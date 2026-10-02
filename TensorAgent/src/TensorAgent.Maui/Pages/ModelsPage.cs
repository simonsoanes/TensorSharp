// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Collections.ObjectModel;
using System.Linq;
using TensorAgent.Core.Catalog;
using TensorAgent.Core.Downloads;
using TensorAgent.Core.Hosting;
using TensorAgent.Core.Settings;
using TensorAgent.Maui.Hosting;

namespace TensorAgent.Maui.Pages;

/// <summary>
/// The built-in model list: what this device can run, what is already on it, and
/// what a download would cost.
///
/// <para>
/// This page has no counterpart in the desktop Web UI, and it cannot have one. The
/// server's model picker lists files an operator put on a disk; here the app is
/// responsible for getting them, which means telling the user the size before they
/// commit to it, resuming an interrupted download rather than starting again, and
/// making a partly-downloaded model obviously partly downloaded.
/// </para>
/// </summary>
public sealed class ModelsPage : ContentPage
{
    private readonly AgentAppHost _app;

    /// <summary>The running app, so the debug reproduction hook can drive the same path a tap does.</summary>
    internal AgentAppHost Host => _app;
    private readonly ObservableCollection<ModelRow> _rows = new();
    // A normal Download tap means "use this when it finishes" only while the user
    // remains on this page and has not chosen another model in the meantime.
    private string? _pendingAutoSelectId;

    /// <summary>
    /// True only while this page is on screen.
    ///
    /// <para>
    /// A download now outlives the page, so a job that finishes while the user is in
    /// the chat must not drag them back here to load a model they may no longer want.
    /// The automatic "downloaded, so use it" step happens only when they are still
    /// looking at the list they started it from.
    /// </para>
    /// </summary>
    private bool _visible;

    public ModelsPage(LoopbackWebHost host)
    {
        _app = host.App;
        Title = "Models";
        BackgroundColor = Theme.Background;
        Padding = new Thickness(0);

        var list = new CollectionView
        {
            ItemsSource = _rows,
            ItemTemplate = new DataTemplate(BuildCell),
            SelectionMode = SelectionMode.None,
            BackgroundColor = Theme.Background,
        };

        Content = new Grid
        {
            BackgroundColor = Theme.Background,
            RowDefinitions = { new RowDefinition(GridLength.Auto), new RowDefinition(GridLength.Star) },
            Children =
            {
                Header(),
                list,
            },
        };
        Grid.SetRow((View)((Grid)Content).Children[1], 1);
    }

    private View Header()
    {
        var label = new Label
        {
            Text = $"Every built-in model, and what each one needs. This device has "
                 + $"{_app.Paths.DeviceMemoryGB} GB, so the ones below the line cannot run here. "
                 + "Downloads resume if interrupted and are kept for next time.",
            TextColor = Theme.Muted,
            FontSize = 13,
            Padding = new Thickness(16, 12, 16, 8),
        };
        return label;
    }

    protected override void OnAppearing()
    {
        base.OnAppearing();
        // Subscribe before taking the snapshot. Otherwise a transfer can complete
        // after Refresh sees Running and before the handler is attached, leaving the
        // reconstructed row stuck forever in that stale state.
        _app.Downloads.Changed -= OnDownloadChanged;
        _app.Downloads.Changed += OnDownloadChanged;
        try
        {
            Refresh();
            _visible = true;
        }
        catch (Exception ex)
        {
            _app.Downloads.Changed -= OnDownloadChanged;
            // A page that cannot list the models is still a page the user reached. Left
            // to propagate, this cancels the push and drops them back on the chat with
            // nothing said -- indistinguishable from the menu not working.
            Console.WriteLine("TensorAgent: the models list failed to appear: " + ex);
        }
    }

    protected override void OnDisappearing()
    {
        base.OnDisappearing();
        _visible = false;
        _pendingAutoSelectId = null;
        _app.Downloads.Changed -= OnDownloadChanged;
    }

    /// <summary>
    /// One report from a running download, from whatever thread the transfer is on.
    ///
    /// <para>
    /// The row is found by id rather than held, because <see cref="Refresh"/> rebuilds
    /// the collection and a captured row would then be updating an object no longer in
    /// the list.
    /// </para>
    /// </summary>
    private void OnDownloadChanged(ModelDownloadStatus status)
    {
        MainThread.BeginInvokeOnMainThread(() =>
        {
            ModelRow? row = _rows.FirstOrDefault(r => string.Equals(r.Model.Id, status.ModelId, StringComparison.Ordinal));
            if (row is null)
                return;

            switch (status.State)
            {
                case DownloadState.Running:
                    row.Report(status.Progress);
                    return;
                case DownloadState.Completed:
                    bool visionOnly = status.RequestsOnly(CatalogFileRole.Projector);
                    bool selectedNow = string.Equals(
                        _app.Settings.Load().SelectedModelId, row.Model.Id, StringComparison.Ordinal);
                    bool autoSelect = !visionOnly && string.Equals(
                        _pendingAutoSelectId, row.Model.Id, StringComparison.Ordinal);
                    if (autoSelect)
                        _pendingAutoSelectId = null;
                    row.Finish(_app.Models);
                    if (_visible && (autoSelect || (visionOnly && selectedNow)))
                        Select(row);
                    else if (_visible && visionOnly && row.IsSelected != selectedNow)
                        Refresh();
                    return;
                case DownloadState.Cancelled:
                    if (string.Equals(_pendingAutoSelectId, row.Model.Id, StringComparison.Ordinal))
                        _pendingAutoSelectId = null;
                    row.Cancelled(_app.Models);
                    return;
                default:
                    if (string.Equals(_pendingAutoSelectId, row.Model.Id, StringComparison.Ordinal))
                        _pendingAutoSelectId = null;
                    row.Failed(_app.Models, status.Error ?? "the download failed");
                    return;
            }
        });
    }

    /// <summary>
    /// Every built-in entry, runnable ones first.
    ///
    /// <para>
    /// This used to list <c>_app.Catalog</c>, which is <c>ForDevice</c> -- only what
    /// fits. That is the right list for LOADING a model and the wrong one for a page
    /// whose job is to tell the user what exists: on a 12 GB iPhone it silently hid
    /// half the catalog, and when a memory-tier bug made ForDevice return nothing the
    /// page went completely blank with no way to tell "none fit" from "something is
    /// broken". A model that needs a bigger device is shown, greyed, saying so.
    /// </para>
    /// </summary>
    private void Refresh()
    {
        string? selected = _app.Settings.Load().SelectedModelId;
        string? loadedModel = _app.ModelService.LoadedModelName;
        bool visionReady = _app.ModelService.Model?.HasVisionEncoder() ?? false;
        int deviceGB = _app.Paths.DeviceMemoryGB;
        _rows.Clear();
        foreach (CatalogModel model in ModelCatalog.BuiltIn
                     .OrderByDescending(m => m.MinDeviceMemoryGB <= deviceGB)
                     .ThenBy(m => m.MinDeviceMemoryGB)
                     .ThenBy(m => m.TotalBytes))
        {
            _rows.Add(new ModelRow(
                model, _app.Models, selected, deviceGB, _app.Downloads.StatusOf(model.Id),
                loadedModel, visionReady));
        }
    }

    private View BuildCell()
    {
        var title = new Label { FontSize = 16, TextColor = Theme.Text, FontAttributes = FontAttributes.Bold };
        title.SetBinding(Label.TextProperty, nameof(ModelRow.Title));

        var subtitle = new Label { FontSize = 12, TextColor = Theme.Muted, LineBreakMode = LineBreakMode.WordWrap };
        subtitle.SetBinding(Label.TextProperty, nameof(ModelRow.Subtitle));

        var status = new Label { FontSize = 12, TextColor = Theme.Accent };
        status.SetBinding(Label.TextProperty, nameof(ModelRow.Status));

        var progress = new ProgressBar { ProgressColor = Theme.Accent, HeightRequest = 3 };
        progress.SetBinding(ProgressBar.ProgressProperty, nameof(ModelRow.Fraction));
        progress.SetBinding(IsVisibleProperty, nameof(ModelRow.IsBusy));

        var action = new Button
        {
            FontSize = 14,
            Padding = new Thickness(14, 6),
            BackgroundColor = Theme.Accent,
            TextColor = Colors.White,
            CornerRadius = 8,
        };
        action.SetBinding(Button.TextProperty, nameof(ModelRow.ActionLabel));
        action.SetBinding(IsEnabledProperty, nameof(ModelRow.Runnable));
        action.SetBinding(Button.BackgroundColorProperty, nameof(ModelRow.ActionColor));
        action.Clicked += (s, _) => OnAction(((Button)s!).BindingContext as ModelRow);

        var addVision = new Button
        {
            Text = "Add vision",
            FontSize = 14,
            Padding = new Thickness(14, 6),
            BackgroundColor = Theme.Accent,
            TextColor = Colors.White,
            CornerRadius = 8,
        };
        addVision.SetBinding(IsVisibleProperty, nameof(ModelRow.CanAddVision));
        addVision.Clicked += (s, _) => OnVisionAction(((Button)s!).BindingContext as ModelRow);

        var remove = new Button
        {
            Text = "Delete",
            FontSize = 14,
            Padding = new Thickness(14, 6),
            BackgroundColor = Theme.Surface,
            TextColor = Theme.Muted,
            CornerRadius = 8,
        };
        remove.SetBinding(IsVisibleProperty, nameof(ModelRow.CanDelete));
        remove.Clicked += (s, _) => OnDelete(((Button)s!).BindingContext as ModelRow);

        var buttons = new HorizontalStackLayout { Spacing = 8, Children = { action, addVision, remove } };

        return new Border
        {
            Margin = new Thickness(12, 6),
            Padding = new Thickness(14),
            BackgroundColor = Theme.Surface,
            StrokeThickness = 0,
            StrokeShape = new Microsoft.Maui.Controls.Shapes.RoundRectangle { CornerRadius = 12 },
            Content = new VerticalStackLayout
            {
                Spacing = 6,
                Children = { title, subtitle, status, progress, buttons },
            },
        };
    }

    private async void OnAction(ModelRow? row)
    {
        if (row is null)
            return;

        if (_app.Downloads.StatusOf(row.Model.Id) is { IsRunning: true })
        {
            _app.Downloads.Cancel(row.Model.Id);
            return;
        }

        // Downloads expose a Stop action above. Loads and local imports do not:
        // accepting a second tap would queue another multi-gigabyte import behind the
        // store lock or race another model load.
        if (row.IsBusy)
            return;

        if (row.IsInstalled)
        {
            Select(row);
            return;
        }

        if (row.Model.SideloadOnly)
        {
            await Import(row);
            return;
        }

        AppSettings settings = _app.Settings.Load();
        if (!settings.AllowCellularDownloads && Services.DeviceState.IsOnCellularOnly())
        {
            await DisplayAlert(
                "Waiting for Wi-Fi",
                $"{row.Model.DisplayName} is {row.Model.TotalBytes / 1e9:0.0} GB and this device is on cellular. "
                + "Turn on \u201CDownload over cellular\u201D in Settings to download it anyway.",
                "OK");
            return;
        }

        Download(row);
    }

    /// <summary>
    /// Copy a publisher-less, hash-pinned card from the Files picker into the model
    /// store. The store stages and verifies the whole file before replacing anything;
    /// selecting the wrong multi-gigabyte GGUF leaves no loadable partial behind.
    /// </summary>
    private async Task Import(ModelRow row)
    {
        try
        {
            FileResult? picked = await FilePicker.Default.PickAsync(new PickOptions
            {
                PickerTitle = $"Choose {row.Model.Weights.FileName}",
            });
            if (picked is null)
                return;

            row.BeginImport();
            await using Stream source = await picked.OpenReadAsync();
            var progress = new Progress<long>(row.ReportImport);
            await _app.Models.ImportAsync(row.Model, source, progress, CancellationToken.None);
            row.Finish(_app.Models);
            Select(row);
        }
        catch (Exception ex)
        {
            row.Failed(_app.Models, ex.Message);
            await DisplayAlert("Could not import this model", ex.Message, "OK");
        }
    }

    /// <summary>
    /// Fetch only the optional projector. The ordinary action remains available for
    /// text-only use, so the global optional-download switch still means what it says;
    /// this second button is explicit consent to add the model's image capability.
    /// </summary>
    private async void OnVisionAction(ModelRow? row)
    {
        if (row is null || !row.NeedsVisionProjector)
            return;

        AppSettings settings = _app.Settings.Load();
        if (!settings.AllowCellularDownloads && Services.DeviceState.IsOnCellularOnly())
        {
            await DisplayAlert(
                "Waiting for Wi-Fi",
                $"The vision file for {row.Model.DisplayName} has "
                + $"{row.VisionBytesRemaining / 1e9:0.0} GB left to download. "
                + "Turn on \u201CDownload over cellular\u201D in Settings to download it anyway.",
                "OK");
            return;
        }

        row.BeginVisionDownload();
        // The draft head rides along when the model lists one and it is not here yet:
        // a model installed before the draft was fetched at all has no other way to
        // get it, and the manager skips files already complete.
        var roles = new List<CatalogFileRole> { CatalogFileRole.Projector };
        if (row.Model.Files.Any(f => f.Role == CatalogFileRole.Draft)
            && _app.Models.CompanionPath(row.Model, CatalogFileRole.Draft) is null)
            roles.Add(CatalogFileRole.Draft);
        _app.Downloads.Start(row.Model, roles);
    }

    /// <summary>
    /// Use this model now, and go back to the chat.
    ///
    /// <para>
    /// This used to save the choice and say it would apply "when TensorAgent next
    /// starts", which on a phone reads as a button that did nothing: the list said the
    /// model was selected while the chat kept answering "No model is configured".
    /// AgentAppHost.UseModel repoints the engine and loads the weights, so the choice
    /// is real by the time this returns.
    /// </para>
    /// <para>
    /// Loading is seconds of work (22 s for a 5 GB model on an iPhone 17 Pro Max), so
    /// it happens off the UI thread with the row showing what it is doing, and the page
    /// returns to the chat by itself afterwards -- being left on the list, having just
    /// chosen something, is a dead end the user has to navigate out of.
    /// </para>
    /// </summary>
    private async void Select(ModelRow row)
    {
        // Choosing any model supersedes a promise to auto-select a different download
        // that happens to finish while this load is in flight.
        _pendingAutoSelectId = null;
        row.BeginLoading();
        try
        {
            string backend = await Task.Run(() => _app.UseModel(row.Model));
            Refresh();
            // Only if this is still the screen the user is looking at. Loading takes
            // twenty seconds and nobody is made to wait here for it: they can go back to
            // the chat, open the drawer and pick another screen while it runs. Popping
            // unconditionally when the load lands takes that screen away again, and from
            // the user's side it looks exactly like a menu item that did nothing.
            if (AppShell.IsOnTop(this))
                await AppShell.BackToChatAsync();
            Console.WriteLine($"TensorAgent: now using {row.Model.Id} on {backend}");
        }
        catch (Exception ex)
        {
            row.Failed(_app.Models, ex.Message);
            await DisplayAlert("Could not use this model", ex.Message, "OK");
            Refresh();
        }
    }

    /// <summary>
    /// Hand the transfer to the download manager and let the row follow it.
    ///
    /// <para>
    /// Nothing is awaited here, and that is the change. This method used to hold the
    /// download for its whole length — the cancellation source lived in this page's
    /// dictionary and the progress went straight into a row — so a user who tapped
    /// Download and went back to the chat took the only owner of a multi-gigabyte
    /// transfer with them. The job is the app's now; this only starts it, and
    /// <see cref="OnDownloadChanged"/> paints whatever it goes on to do.
    /// </para>
    /// </summary>
    private void Download(ModelRow row)
    {
        _pendingAutoSelectId = row.Model.Id;
        row.BeginDownload();
        _app.Downloads.Start(
            row.Model,
            ModelDownloadManager.OptionalRolesFor(_app.Settings.Load().DownloadOptionalFiles));
    }

    private async void OnDelete(ModelRow? row)
    {
        if (row is null)
            return;
        string recovery = row.Model.SideloadOnly
            ? "You will need to choose the original local GGUF again to restore it."
            : "It can be downloaded again.";
        if (!await DisplayAlert("Delete model",
                $"Remove {row.Model.DisplayName} from this device? {recovery}", "Delete", "Cancel"))
        {
            return;
        }
        _app.DeleteModel(row.Model);
        Refresh();
    }
}

/// <summary>One row of the model list, and the only place its display state lives.</summary>
public sealed class ModelRow : BindableObject
{
    private string _status;
    private double _fraction;
    private bool _busy;
    private string _actionLabel;

    /// <param name="download">
    /// What this launch's download manager is doing with the entry, or null when it has
    /// never touched it. It is a constructor parameter because the page is rebuilt every
    /// time it appears, and a row built without it shows "Partly downloaded · 3.1 GB
    /// still to fetch" beside a Download button for a transfer that is running right
    /// now — the one state the user must not be invited to start again.
    /// </param>
    public ModelRow(
        CatalogModel model, ModelStore store, string? selectedId, int deviceMemoryGB,
        ModelDownloadStatus? download = null, string? loadedModelName = null,
        bool loadedVisionReady = false)
    {
        Model = model;
        Runnable = model.MinDeviceMemoryGB <= deviceMemoryGB;
        DeviceMemoryGB = deviceMemoryGB;
        RefreshInstallState(store);
        IsSelected = string.Equals(model.Id, selectedId, StringComparison.Ordinal);
        VisionActivationRequired = IsSelected
            && IsInstalled
            && model.Modalities.HasFlag(CatalogModalities.Image)
            && store.CompanionPath(model, CatalogFileRole.Projector) is not null
            && string.Equals(model.Weights.FileName, loadedModelName, StringComparison.OrdinalIgnoreCase)
            && !loadedVisionReady;
        _status = DescribeState(store);
        _actionLabel = !Runnable ? "Too big"
            : VisionActivationRequired ? "Enable vision"
            : IsInstalled ? (IsSelected ? "Selected" : "Use")
            : model.SideloadOnly ? "Import"
            : "Download";

        if (download is { IsRunning: true } running)
        {
            BeginDownload();
            Report(running.Progress);
        }
        else if (download is { State: DownloadState.Failed } failed)
        {
            Failed(store, failed.Error ?? "the download failed");
        }
        else if (download is { State: DownloadState.Cancelled } && !IsInstalled)
        {
            Cancelled(store);
        }
    }

    public CatalogModel Model { get; }

    /// <summary>Whether this device has the memory the entry asks for.</summary>
    public bool Runnable { get; }

    public int DeviceMemoryGB { get; }
    public bool IsInstalled { get; private set; }
    public bool IsSelected { get; }
    /// <summary>Weights can answer text, but this advertised image model lacks its optional projector.</summary>
    public bool NeedsVisionProjector { get; private set; }
    /// <summary>Bytes left in the projector download, accounting for a resumable .part.</summary>
    public long VisionBytesRemaining { get; private set; }
    /// <summary>The projector arrived after this selected model was loaded text-only.</summary>
    public bool VisionActivationRequired { get; }

    public string Title => Model.DisplayName + (IsSelected ? "  ·  in use" : string.Empty);

    /// <summary>
    /// What the model IS, in the order someone deciding actually asks: how big is it,
    /// what can it take in, and what is it for. The description used to appear only on
    /// experimental entries, so most of the list was a size and nothing else.
    /// </summary>
    public string Subtitle =>
        $"{Model.Parameters} · {Model.Quantization} · {Gb(Model.TotalBytes)} GB "
        + (Model.SideloadOnly ? "local file" : "download")
        + (Model.Kind == CatalogArchitectureKind.MixtureOfExperts ? " · mixture of experts" : string.Empty)
        + $"\nReads: {Reads}"
        + (Model.SupportsThinking ? " · thinks when asked" : string.Empty)
        + (string.IsNullOrWhiteSpace(Model.Notes) ? string.Empty
            : "\n" + (Model.Experimental ? "Experimental: " : string.Empty) + Model.Notes);

    /// <summary>Every input this entry accepts, not just images, and for a model that makes
    /// pictures or clips rather than text, what it makes.</summary>
    private string Reads
    {
        get
        {
            var parts = new List<string> { "text" };
            if (Model.Modalities.HasFlag(CatalogModalities.Image)) parts.Add("images");
            if (Model.Modalities.HasFlag(CatalogModalities.Audio)) parts.Add("audio");
            if (Model.Modalities.HasFlag(CatalogModalities.Video)) parts.Add("video");
            if (Model.Kind != CatalogArchitectureKind.Diffusion) return string.Join(", ", parts);
            string makes = Model.IsVideoGenerator
                ? Model.Modalities.HasFlag(CatalogModalities.AudioOutput) ? "makes video clips with sound" : "makes video clips"
                : "makes pictures";
            return parts.Count == 1 ? $"a prompt, and {makes}" : $"a prompt and {string.Join(", ", parts.Skip(1))}, and {makes}";
        }
    }

    /// <summary>Greyed out when the device cannot run it, so the button reads as inert.</summary>
    public Color ActionColor => Runnable ? Theme.Accent : Theme.Surface;

    public string Status { get => _status; private set { _status = value; OnPropertyChanged(); } }
    public double Fraction { get => _fraction; private set { _fraction = value; OnPropertyChanged(); } }
    public bool IsBusy { get => _busy; private set { _busy = value; OnPropertyChanged(); OnPropertyChanged(nameof(CanDelete)); OnPropertyChanged(nameof(CanAddVision)); } }
    public string ActionLabel { get => _actionLabel; private set { _actionLabel = value; OnPropertyChanged(); } }
    public bool CanDelete => IsInstalled && !IsBusy;
    public bool CanAddVision => Runnable && NeedsVisionProjector && !IsBusy;

    public void BeginDownload()
    {
        IsBusy = true;
        ActionLabel = "Stop";
        Status = "Starting…";
    }

    public void BeginVisionDownload()
    {
        IsBusy = true;
        ActionLabel = "Stop";
        Status = "Starting the vision download…";
    }

    public void BeginImport()
    {
        IsBusy = true;
        ActionLabel = "Importing…";
        Status = "Copying and verifying the local GGUF…";
    }

    public void ReportImport(long bytes)
    {
        Fraction = Model.TotalBytes > 0 ? Math.Min(1.0, (double)bytes / Model.TotalBytes) : 0;
        Status = $"Importing · {Gb(bytes)}/{Gb(Model.TotalBytes)} GB";
    }

    /// <summary>Loading the weights, which is seconds rather than instant.</summary>
    public void BeginLoading()
    {
        IsBusy = true;
        ActionLabel = "Loading…";
        Status = "Loading onto the GPU…";
    }

    public void Report(ModelDownloadProgress p)
    {
        Fraction = p.Fraction;
        string speed = p.BytesPerSecond > 1 ? $" · {p.BytesPerSecond / 1e6:0.0} MB/s" : string.Empty;
        string eta = p.Eta is { } left ? $" · {Math.Round(left.TotalMinutes)} min left" : string.Empty;
        // FileIndex is already 1-based (ModelDownloadProgress); adding one more showed the
        // first of seven files as "2/7" and a finished download as "8/7".
        Status = $"{p.Phase} {p.FileIndex}/{p.FileCount} · {Gb(p.BytesReceived)}/{Gb(p.TotalBytes)} GB{speed}{eta}";
    }

    public void Finish(ModelStore store)
    {
        IsBusy = false;
        RefreshInstallState(store);
        Fraction = 1;
        ActionLabel = IsSelected ? "Selected" : "Use";
        Status = DescribeState(store);
        OnPropertyChanged(nameof(CanDelete));
        OnPropertyChanged(nameof(CanAddVision));
    }

    public void Cancelled(ModelStore store)
    {
        IsBusy = false;
        RefreshInstallState(store);
        ActionLabel = IsInstalled
            ? (VisionActivationRequired ? "Enable vision" : IsSelected ? "Selected" : "Use")
            : "Resume";
        Status = "Stopped. " + DescribeState(store);
        OnPropertyChanged(nameof(CanAddVision));
    }

    public void Failed(ModelStore store, string message)
    {
        IsBusy = false;
        RefreshInstallState(store);
        ActionLabel = IsInstalled
            ? (VisionActivationRequired ? "Enable vision" : IsSelected ? "Selected" : "Use")
            : Model.SideloadOnly ? "Import"
            : "Retry";
        Status = message;
        OnPropertyChanged(nameof(CanAddVision));
    }

    private string DescribeState(ModelStore store)
    {
        if (!Runnable)
        {
            // Said as a fact about the hardware rather than as a refusal, and it names
            // both numbers so the user can see how far off it is instead of guessing.
            return $"Needs a {Model.MinDeviceMemoryGB} GB device · this one has {DeviceMemoryGB} GB";
        }

        if (NeedsVisionProjector)
        {
            return $"Text ready · vision file not downloaded · {Gb(VisionBytesRemaining)} GB to add · {Model.License}";
        }

        if (VisionActivationRequired)
        {
            return $"Vision downloaded · tap Enable vision to load it · {Model.License}";
        }

        if (Model.SideloadOnly && store.StateOf(Model) != InstallState.Installed)
        {
            return $"Not imported · choose {Model.Weights.FileName} from Files · {Model.License}";
        }

        return store.StateOf(Model) switch
        {
            InstallState.Installed => $"On this device · {Gb(store.InstalledBytes(Model))} GB · {Model.License}",
            InstallState.Partial => $"Partly downloaded · {Gb(store.RemainingBytes(Model))} GB still to fetch",
            // What the download will actually move: files another installed entry already
            // holds are linked, not fetched (the second MiniMax-H3 checkpoint is its 11 GB
            // denoiser, not 35 GB).
            _ when store.RemainingBytes(Model) is long fetch && fetch < Model.TotalBytes =>
                $"Not downloaded · {Gb(fetch)} GB to fetch, the rest already on this device · {Model.License}",
            _ => $"Not downloaded · {Gb(Model.TotalBytes)} GB · {Model.License}",
        };
    }

    private void RefreshInstallState(ModelStore store)
    {
        IsInstalled = store.StateOf(Model) == InstallState.Installed;
        CatalogFile? projector = Model.Projector;
        NeedsVisionProjector = IsInstalled
            && projector is { Optional: true }
            && Model.Modalities.HasFlag(CatalogModalities.Image)
            && store.CompanionPath(Model, CatalogFileRole.Projector) is null;

        VisionBytesRemaining = 0;
        if (!NeedsVisionProjector || projector is null)
            return;

        string part = ResumableDownloader.PartPath(store.PathFor(Model, projector));
        long have = File.Exists(part) ? Math.Min(new FileInfo(part).Length, projector.Bytes) : 0;
        VisionBytesRemaining = projector.Bytes - have;
    }

    private static string Gb(long bytes) => (bytes / 1e9).ToString("0.00");
}

/// <summary>The Web UI's own palette, so the native pages do not look bolted on.</summary>
internal static class Theme
{
    public static readonly Color Background = Color.FromArgb("#0b1220");
    public static readonly Color Surface = Color.FromArgb("#151d31");
    public static readonly Color Text = Color.FromArgb("#e6edf7");
    public static readonly Color Muted = Color.FromArgb("#8b9ab8");
    public static readonly Color Accent = Color.FromArgb("#3b82f6");
    public static readonly Color Danger = Color.FromArgb("#ef4444");
}
