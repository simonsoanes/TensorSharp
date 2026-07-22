using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Threading;

namespace TensorSharp.Distributed
{
    /// <summary>
    /// Collective operations over a TCP mesh network. Each node knows its own
    /// rank and the endpoints of every other node. Connections are established
    /// once at construction and reused for all subsequent operations.
    ///
    /// Wire protocol (per message):
    ///   [4 bytes] payload length (little-endian int32, excludes this header)
    ///   [1 byte ] message type
    ///   [N bytes] payload
    ///
    /// AllReduce algorithm (reduce-to-zero + broadcast):
    ///   1. Every non-zero node sends its partial to node 0
    ///   2. Node 0 accumulates: sum += partial
    ///   3. Node 0 broadcasts the reduced result to every other node
    ///   4. Barrier so all nodes observe the result before proceeding
    ///
    /// This mirrors the local <c>CudaP2PCommunicator</c> algorithm and
    /// preserves the bitwise-determinism invariant: because node 0 sums in
    /// a fixed rank order, every node receives an identical result.
    /// </summary>
    public sealed class TcpCommunicator : IDisposable
    {
        private const byte MsgAllReduceContribute = 0x01;
        private const byte MsgAllReduceResult     = 0x02;
        private const byte MsgBarrierAck          = 0x03;

        private readonly int _rank;
        private readonly int _worldSize;
        private readonly TcpListener _listener;
        private readonly TcpClient[] _peers;   // indexed by remote rank
        private readonly NetworkStream[] _streams;
        private readonly object _sendLock = new();
        private bool _disposed;

        // Reusable receive buffer to avoid per-call allocation.
        private byte[] _recvBuffer = Array.Empty<byte>();

        /// <param name="rank">This node's rank (0..worldSize-1).</param>
        /// <param name="endpoints">
        /// Endpoints for every node in the group, indexed by rank.
        /// The entry at <paramref name="rank"/> is the local listen address.
        /// </param>
        public TcpCommunicator(int rank, IPEndPoint[] endpoints)
        {
            _rank = rank;
            _worldSize = endpoints.Length;

            if (_worldSize < 2)
                throw new ArgumentException("TCP communicator requires at least 2 nodes.", nameof(endpoints));
            if ((uint)rank >= (uint)_worldSize)
                throw new ArgumentOutOfRangeException(nameof(rank));

            _peers = new TcpClient[_worldSize];
            _streams = new NetworkStream[_worldSize];

            // Start listening.
            _listener = new TcpListener(endpoints[rank]);
            _listener.Start();

            try
            {
                // Lower-ranked node initiates the connection to higher-ranked nodes.
                // Higher-ranked nodes accept. This gives exactly one connection per pair.
                var acceptFutures = new List<(int remoteRank, TcpClient client)>();

                // Initiate outbound connections to higher-ranked peers.
                for (int r = rank + 1; r < _worldSize; r++)
                {
                    var client = new TcpClient { NoDelay = true };
                    client.Connect(endpoints[r]);
                    _peers[r] = client;
                    _streams[r] = client.GetStream();
                }

                // Accept inbound connections from lower-ranked peers.
                for (int r = 0; r < rank; r++)
                {
                    var client = _listener.AcceptTcpClient();
                    client.NoDelay = true;
                    _peers[r] = client;
                    _streams[r] = client.GetStream();
                }
            }
            catch
            {
                Dispose();
                throw;
            }

            Console.WriteLine($"[TcpCommunicator] Rank {rank}/{_worldSize} connected to all peers.");
        }

        public int Rank => _rank;
        public int WorldSize => _worldSize;

        /// <summary>
        /// In-place AllReduce (sum) across all nodes. The buffer is modified
        /// to contain the element-wise sum of all nodes' inputs.
        /// All nodes must call this concurrently with the same buffer length.
        /// </summary>
        public unsafe void AllReduce(Span<float> buffer)
        {
            int byteCount = buffer.Length * sizeof(float);

            if (_rank == 0)
            {
                // Node 0: receive partials from all other nodes and accumulate.
                for (int r = 1; r < _worldSize; r++)
                {
                    var (msgType, payload) = ReceiveMessage(r);
                    if (msgType != MsgAllReduceContribute)
                        throw new InvalidOperationException(
                            $"Expected AllReduceContribute from rank {r}, got message type 0x{msgType:X2}.");

                    fixed (float* bufPtr = buffer)
                    fixed (byte* payPtr = payload)
                    {
                        float* src = (float*)payPtr;
                        for (int i = 0; i < buffer.Length; i++)
                            bufPtr[i] += src[i];
                    }
                }

                // Broadcast the reduced result to all other nodes.
                byte[] resultBytes = new byte[byteCount];
                fixed (float* bufPtr = buffer)
                    System.Runtime.InteropServices.Marshal.Copy((IntPtr)bufPtr, resultBytes, 0, byteCount);

                for (int r = 1; r < _worldSize; r++)
                    SendMessage(r, MsgAllReduceResult, resultBytes);
            }
            else
            {
                // Non-zero node: send partial to node 0, then receive the result.
                byte[] partialBytes = new byte[byteCount];
                fixed (float* bufPtr = buffer)
                    System.Runtime.InteropServices.Marshal.Copy((IntPtr)bufPtr, partialBytes, 0, byteCount);

                SendMessage(0, MsgAllReduceContribute, partialBytes);

                var (msgType, payload) = ReceiveMessage(0);
                if (msgType != MsgAllReduceResult)
                    throw new InvalidOperationException(
                        $"Expected AllReduceResult from rank 0, got message type 0x{msgType:X2}.");

                fixed (float* bufPtr = buffer)
                    System.Runtime.InteropServices.Marshal.Copy(payload, 0, (IntPtr)bufPtr, buffer.Length);
            }

            // Barrier so all nodes have the result before any proceeds.
            Barrier();
        }

        /// <summary>
        /// Broadcast a buffer from <paramref name="rootRank"/> to all nodes.
        /// </summary>
        public unsafe void Broadcast(Span<float> buffer, int rootRank)
        {
            int byteCount = buffer.Length * sizeof(float);

            if (_rank == rootRank)
            {
                byte[] data = new byte[byteCount];
                fixed (float* bufPtr = buffer)
                    System.Runtime.InteropServices.Marshal.Copy((IntPtr)bufPtr, data, 0, byteCount);

                for (int r = 0; r < _worldSize; r++)
                {
                    if (r == rootRank) continue;
                    SendMessage(r, MsgAllReduceResult, data);
                }
            }
            else
            {
                var (msgType, payload) = ReceiveMessage(rootRank);
                if (msgType != MsgAllReduceResult)
                    throw new InvalidOperationException(
                        $"Expected Broadcast from rank {rootRank}, got message type 0x{msgType:X2}.");

                fixed (float* bufPtr = buffer)
                    System.Runtime.InteropServices.Marshal.Copy(payload, 0, (IntPtr)bufPtr, buffer.Length);
            }

            Barrier();
        }

        /// <summary>
        /// Block until all nodes have reached this point. Uses a simple
        /// gather-then-release protocol through node 0.
        /// </summary>
        public void Barrier()
        {
            if (_rank == 0)
            {
                // Receive ack from every other node.
                for (int r = 1; r < _worldSize; r++)
                {
                    var (msgType, _) = ReceiveMessage(r);
                    if (msgType != MsgBarrierAck)
                        throw new InvalidOperationException(
                            $"Expected BarrierAck from rank {r}, got 0x{msgType:X2}.");
                }

                // Release all nodes.
                for (int r = 1; r < _worldSize; r++)
                    SendMessage(r, MsgBarrierAck, Array.Empty<byte>());
            }
            else
            {
                SendMessage(0, MsgBarrierAck, Array.Empty<byte>());
                var (msgType, _) = ReceiveMessage(0);
                if (msgType != MsgBarrierAck)
                    throw new InvalidOperationException(
                        $"Expected BarrierAck from rank 0, got 0x{msgType:X2}.");
            }
        }

        private void SendMessage(int destRank, byte msgType, byte[] payload)
        {
            var stream = _streams[destRank];
            int totalLen = 1 + payload.Length;

            byte[] header = new byte[4];
            BinaryPrimitives.WriteInt32LittleEndian(header, totalLen);

            lock (_sendLock)
            {
                stream.Write(header, 0, 4);
                stream.WriteByte(msgType);
                if (payload.Length > 0)
                    stream.Write(payload, 0, payload.Length);
                stream.Flush();
            }
        }

        private (byte msgType, byte[] payload) ReceiveMessage(int srcRank)
        {
            var stream = _streams[srcRank];

            // Read 4-byte length header.
            byte[] header = ReadExact(stream, 4);
            int payloadLen = BinaryPrimitives.ReadInt32LittleEndian(header);

            if (payloadLen < 1)
                throw new InvalidOperationException("Received message with empty payload (missing type byte).");

            // Read type byte + payload.
            byte[] full = ReadExact(stream, payloadLen);
            byte msgType = full[0];
            byte[] payload = new byte[payloadLen - 1];
            if (payload.Length > 0)
                Array.Copy(full, 1, payload, 0, payload.Length);

            return (msgType, payload);
        }

        private static byte[] ReadExact(NetworkStream stream, int count)
        {
            byte[] buf = new byte[count];
            int offset = 0;
            while (offset < count)
            {
                int n = stream.Read(buf, offset, count - offset);
                if (n == 0)
                    throw new EndOfStreamException("Peer closed the connection.");
                offset += n;
            }
            return buf;
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;

            for (int r = 0; r < _worldSize; r++)
            {
                _streams[r]?.Dispose();
                _peers[r]?.Dispose();
            }

            try { _listener?.Stop(); } catch { }
        }
    }
}
