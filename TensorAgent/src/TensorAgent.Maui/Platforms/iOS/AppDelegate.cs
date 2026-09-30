// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using Foundation;
using TensorAgent.Core.Hosting;
using TensorAgent.Maui.Services;
using UIKit;

namespace TensorAgent.Maui;

[Register("AppDelegate")]
public class AppDelegate : MauiUIApplicationDelegate
{
    protected override MauiApp CreateMauiApp() => MauiProgram.CreateMauiApp();

    /// <summary>
    /// iOS asking for memory back is the last warning before jetsam, and the app used
    /// to only write it down.
    ///
    /// <para>
    /// The record stays -- a jetsam kill produces no stack, no exception and no message
    /// of its own, so the warning and the headroom at that moment are the last thing
    /// written -- and now the host is asked to give back what it can:
    /// <see cref="AgentAppHost.RelieveMemoryPressure"/>. That is not the weights (a
    /// mapping the engine is reading) nor the cache of the turn in progress (mid-token),
    /// but everything the engine keeps only for the NEXT request's speed: finished
    /// conversations' caches beyond the newest, holders parked for reuse, the pool's
    /// spare host blocks, and the managed heap a long agentic turn leaves behind. The
    /// engine frees them on its own thread between steps; this only asks.
    /// </para>
    /// </summary>
    [Export("applicationDidReceiveMemoryWarning:")]
    public void DidReceiveMemoryWarning(UIApplication application)
    {
        Console.WriteLine($"TensorAgent: iOS memory warning -- {DeviceState.DescribeMemory()}");
        try
        {
            Services?.GetService<Hosting.LoopbackWebHost>()?.App.RelieveMemoryPressure();
        }
        catch (Exception ex)
        {
            // The warning must never be what crashes the app it is trying to save.
            Console.WriteLine($"TensorAgent: relieving memory pressure failed: {ex.Message}");
        }
    }
}
