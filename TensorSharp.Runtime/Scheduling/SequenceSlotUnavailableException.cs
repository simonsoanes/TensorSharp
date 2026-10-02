// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;

namespace TensorSharp.Runtime.Scheduling
{
    /// <summary>
    /// A model that keeps every concurrent request in its own device-resident slot (DeepSeek V4.1 and
    /// GLM 5.x native executors) could not allocate one for a new request: device memory holds no more.
    /// This is a capacity limit, not a failure of the request. The executor reports it
    /// (<see cref="SequenceStepResult.SlotUnavailable"/>), and the scheduler puts the request back at the
    /// front of the waiting queue and admits nothing past the requests that hold slots until one of them
    /// finishes. Thrown before anything about the request's state changed.
    /// </summary>
    public sealed class SequenceSlotUnavailableException : InvalidOperationException
    {
        public SequenceSlotUnavailableException(string message) : base(message) { }
    }
}
