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
using TensorAgent.Maui.Hosting;
using UIKit;

namespace TensorAgent.Maui;

/// <summary>
/// The Mac app's delegate. The phone's delegate also answers memory warnings, which
/// exist there because iOS kills an app over its budget without a word; macOS pages
/// instead, so there is nothing for the Mac app to give back on request.
/// </summary>
[Register("AppDelegate")]
public class AppDelegate : MauiUIApplicationDelegate
{
    protected override MauiApp CreateMauiApp() => MauiProgram.CreateMauiApp();

    /// <summary>
    /// The one notice a quitting Mac app gets before <c>exit()</c>: release the model and
    /// the engine now, or ggml-metal's static destructor aborts the process on the buffers
    /// still registered (see <see cref="LoopbackWebHost.ShutDownForTermination"/>).
    /// </summary>
    public override void WillTerminate(UIApplication application)
    {
        LoopbackWebHost.ShutDownForTermination();
        base.WillTerminate(application);
    }
}
