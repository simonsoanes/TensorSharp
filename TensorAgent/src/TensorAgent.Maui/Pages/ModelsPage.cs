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
using TensorAgent.Core.Localization;
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
        BackgroundColor = Theme.Background;
        Padding = new Thickness(0);
        Build();

        // A new language rebuilds the screen, cells and all. The rows are rebuilt in it by
        // the next Refresh, which appearing always runs.
        Loc.Changed += () => Dispatcher.Dispatch(() =>
        {
            try
            {
                Build();
                if (_visible)
                    Refresh();
            }
            catch (Exception ex)
            {
                Console.WriteLine("TensorAgent: the models list failed to repaint in the new language: " + ex);
            }
        });
    }

    private void Build()
    {
        Title = Loc.T("models.title");

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
            Text = Loc.T("models.header", ("memory", _app.Paths.DeviceMemoryGB)),
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
                    row.Failed(_app.Models, status.Error ?? Loc.T("models.status.downloadFailed"));
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
            Text = Loc.T("models.action.addVision"),
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
            Text = Loc.T("models.action.delete"),
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
                Loc.T("models.alert.cellular.title"),
                Loc.T("models.alert.cellular.model",
                    ("model", row.Model.DisplayName),
                    ("size", (row.Model.TotalBytes / 1e9).ToString("0.0", Loc.Culture)),
                    ("setting", Loc.T("settings.downloads.cellular.title"))),
                Loc.T("common.ok"));
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
                PickerTitle = Loc.T("models.import.pickerTitle", ("file", row.Model.Weights.FileName)),
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
            await DisplayAlert(Loc.T("models.alert.importFailed.title"), ex.Message, Loc.T("common.ok"));
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
                Loc.T("models.alert.cellular.title"),
                Loc.T("models.alert.cellular.vision",
                    ("model", row.Model.DisplayName),
                    ("size", (row.VisionBytesRemaining / 1e9).ToString("0.0", Loc.Culture)),
                    ("setting", Loc.T("settings.downloads.cellular.title"))),
                Loc.T("common.ok"));
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
            await DisplayAlert(Loc.T("models.alert.useFailed.title"), ex.Message, Loc.T("common.ok"));
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
        string message = row.Model.SideloadOnly
            ? Loc.T("models.alert.delete.messageLocal", ("model", row.Model.DisplayName))
            : Loc.T("models.alert.delete.message", ("model", row.Model.DisplayName));
        if (!await DisplayAlert(Loc.T("models.alert.delete.title"), message,
                Loc.T("models.alert.delete.confirm"), Loc.T("common.cancel")))
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
        _actionLabel = !Runnable ? Loc.T("models.action.tooBig")
            : VisionActivationRequired ? Loc.T("models.action.enableVision")
            : IsInstalled ? (IsSelected ? Loc.T("models.action.selected") : Loc.T("models.action.use"))
            : model.SideloadOnly ? Loc.T("models.action.import")
            : Loc.T("models.action.download");

        if (download is { IsRunning: true } running)
        {
            BeginDownload();
            Report(running.Progress);
        }
        else if (download is { State: DownloadState.Failed } failed)
        {
            Failed(store, failed.Error ?? Loc.T("models.status.downloadFailed"));
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

    public string Title => IsSelected ? Loc.T("models.row.inUse", ("model", Model.DisplayName)) : Model.DisplayName;

    /// <summary>
    /// What the model IS, in the order someone deciding actually asks: how big is it,
    /// what can it take in, and what is it for. The description used to appear only on
    /// experimental entries, so most of the list was a size and nothing else.
    /// </summary>
    public string Subtitle =>
        $"{Model.Parameters} · {Model.Quantization} · "
        + (Model.SideloadOnly
            ? Loc.T("models.row.localFileSize", ("size", Gb(Model.TotalBytes)))
            : Loc.T("models.row.downloadSize", ("size", Gb(Model.TotalBytes))))
        + (Model.Kind == CatalogArchitectureKind.MixtureOfExperts ? " · " + Loc.T("models.row.mixtureOfExperts") : string.Empty)
        + "\n" + Reads
        + (Model.SupportsThinking ? " · " + Loc.T("models.row.thinks") : string.Empty)
        + (string.IsNullOrWhiteSpace(Model.Notes) ? string.Empty
            : "\n" + (Model.Experimental ? Loc.T("models.row.experimental", ("notes", Model.Notes)) : Model.Notes));

    /// <summary>Every input this entry accepts, not just images, and for a model that makes
    /// pictures or clips rather than text, what it makes.</summary>
    private string Reads
    {
        get
        {
            var parts = new List<string> { Loc.T("models.row.input.text") };
            if (Model.Modalities.HasFlag(CatalogModalities.Image)) parts.Add(Loc.T("models.row.input.images"));
            if (Model.Modalities.HasFlag(CatalogModalities.Audio)) parts.Add(Loc.T("models.row.input.audio"));
            if (Model.Modalities.HasFlag(CatalogModalities.Video)) parts.Add(Loc.T("models.row.input.video"));
            string separator = Loc.T("models.row.input.separator");
            if (Model.Kind != CatalogArchitectureKind.Diffusion)
                return Loc.T("models.row.reads", ("inputs", string.Join(separator, parts)));
            // What it reads and what it makes is one sentence, so each kind is a whole line,
            // with and without inputs beyond the prompt.
            string inputs = string.Join(separator, parts.Skip(1));
            if (!Model.IsVideoGenerator)
            {
                return parts.Count == 1
                    ? Loc.T("models.row.makes.pictures")
                    : Loc.T("models.row.makes.picturesWithInputs", ("inputs", inputs));
            }
            if (Model.Modalities.HasFlag(CatalogModalities.AudioOutput))
            {
                return parts.Count == 1
                    ? Loc.T("models.row.makes.clipsWithSound")
                    : Loc.T("models.row.makes.clipsWithSoundWithInputs", ("inputs", inputs));
            }
            return parts.Count == 1
                ? Loc.T("models.row.makes.clips")
                : Loc.T("models.row.makes.clipsWithInputs", ("inputs", inputs));
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
        ActionLabel = Loc.T("models.action.stop");
        Status = Loc.T("models.status.starting");
    }

    public void BeginVisionDownload()
    {
        IsBusy = true;
        ActionLabel = Loc.T("models.action.stop");
        Status = Loc.T("models.status.startingVision");
    }

    public void BeginImport()
    {
        IsBusy = true;
        ActionLabel = Loc.T("models.action.importing");
        Status = Loc.T("models.status.importing");
    }

    public void ReportImport(long bytes)
    {
        Fraction = Model.TotalBytes > 0 ? Math.Min(1.0, (double)bytes / Model.TotalBytes) : 0;
        Status = Loc.T("models.status.importProgress", ("copied", Gb(bytes)), ("total", Gb(Model.TotalBytes)));
    }

    /// <summary>Loading the weights, which is seconds rather than instant.</summary>
    public void BeginLoading()
    {
        IsBusy = true;
        ActionLabel = Loc.T("models.action.loading");
        Status = Loc.T("models.status.loading");
    }

    public void Report(ModelDownloadProgress p)
    {
        Fraction = p.Fraction;
        // FileIndex is already 1-based (ModelDownloadProgress); adding one more showed the
        // first of seven files as "2/7" and a finished download as "8/7".
        var parts = new List<string>
        {
            p.Phase == "verifying"
                ? Loc.T("models.progress.verifying", ("file", p.FileIndex), ("files", p.FileCount))
                : Loc.T("models.progress.downloading", ("file", p.FileIndex), ("files", p.FileCount)),
            Loc.T("models.progress.size", ("received", Gb(p.BytesReceived)), ("total", Gb(p.TotalBytes))),
        };
        if (p.BytesPerSecond > 1)
            parts.Add(Loc.T("models.progress.speed", ("speed", (p.BytesPerSecond / 1e6).ToString("0.0", Loc.Culture))));
        if (p.Eta is { } left)
            parts.Add(Loc.T("models.progress.eta", ("minutes", Math.Round(left.TotalMinutes))));
        Status = string.Join(" · ", parts);
    }

    public void Finish(ModelStore store)
    {
        IsBusy = false;
        RefreshInstallState(store);
        Fraction = 1;
        ActionLabel = IsSelected ? Loc.T("models.action.selected") : Loc.T("models.action.use");
        Status = DescribeState(store);
        OnPropertyChanged(nameof(CanDelete));
        OnPropertyChanged(nameof(CanAddVision));
    }

    public void Cancelled(ModelStore store)
    {
        IsBusy = false;
        RefreshInstallState(store);
        ActionLabel = IsInstalled
            ? (VisionActivationRequired ? Loc.T("models.action.enableVision") : IsSelected ? Loc.T("models.action.selected") : Loc.T("models.action.use"))
            : Loc.T("models.action.resume");
        Status = Loc.T("models.status.stopped", ("state", DescribeState(store)));
        OnPropertyChanged(nameof(CanAddVision));
    }

    public void Failed(ModelStore store, string message)
    {
        IsBusy = false;
        RefreshInstallState(store);
        ActionLabel = IsInstalled
            ? (VisionActivationRequired ? Loc.T("models.action.enableVision") : IsSelected ? Loc.T("models.action.selected") : Loc.T("models.action.use"))
            : Model.SideloadOnly ? Loc.T("models.action.import")
            : Loc.T("models.action.retry");
        Status = message;
        OnPropertyChanged(nameof(CanAddVision));
    }

    private string DescribeState(ModelStore store)
    {
        if (!Runnable)
        {
            // Said as a fact about the hardware rather than as a refusal, and it names
            // both numbers so the user can see how far off it is instead of guessing.
            return Loc.T("models.status.needsDevice", ("required", Model.MinDeviceMemoryGB), ("available", DeviceMemoryGB));
        }

        if (NeedsVisionProjector)
        {
            return Loc.T("models.status.visionMissing", ("size", Gb(VisionBytesRemaining)), ("license", Model.License));
        }

        if (VisionActivationRequired)
        {
            return Loc.T("models.status.visionDownloaded", ("action", Loc.T("models.action.enableVision")), ("license", Model.License));
        }

        if (Model.SideloadOnly && store.StateOf(Model) != InstallState.Installed)
        {
            return Loc.T("models.status.notImported", ("file", Model.Weights.FileName), ("license", Model.License));
        }

        return store.StateOf(Model) switch
        {
            InstallState.Installed => Loc.T("models.status.installed", ("size", Gb(store.InstalledBytes(Model))), ("license", Model.License)),
            InstallState.Partial => Loc.T("models.status.partial", ("size", Gb(store.RemainingBytes(Model)))),
            // What the download will actually move: files another installed entry already
            // holds are linked, not fetched (the second MiniMax-H3 checkpoint is its 11 GB
            // denoiser, not 35 GB).
            _ when store.RemainingBytes(Model) is long fetch && fetch < Model.TotalBytes =>
                Loc.T("models.status.notDownloadedShared", ("size", Gb(fetch)), ("license", Model.License)),
            _ => Loc.T("models.status.notDownloaded", ("size", Gb(Model.TotalBytes)), ("license", Model.License)),
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

    private static string Gb(long bytes) => (bytes / 1e9).ToString("0.00", Loc.Culture);
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
