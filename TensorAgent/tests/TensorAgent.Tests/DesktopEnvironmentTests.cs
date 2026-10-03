using TensorAgent.Core.Hosting;

namespace TensorAgent.Tests;

/// <summary>
/// The PATH a Mac app launched from the Finder has to borrow from the login shell.
///
/// <para>
/// launchd gives a GUI app <c>/usr/bin:/bin:/usr/sbin:/sbin</c>, so Homebrew's node, npm
/// and python3.13 -- what the desktop backend runs skills with -- are not found at all
/// and the Playwright skill fails on its first step, while the same build started from
/// a terminal works. <see cref="DesktopEnvironment"/> merges the login shell's PATH in
/// front of the app's own.
/// </para>
/// </summary>
[Collection(ProcessEnvironmentCollection.Name)]
public sealed class DesktopEnvironmentTests
{
    private static readonly string Sep = Path.PathSeparator.ToString();

    private static string Join(params string[] entries) => string.Join(Sep, entries);

    [Fact]
    public void TheLoginShellsEntriesComeFirstAndEachDirectoryAppearsOnce()
    {
        string merged = DesktopEnvironment.MergePath(
            Join("/opt/homebrew/bin", "/usr/bin", ""),
            new[] { "/usr/local/bin" },
            Join("/usr/bin", "/bin", "/opt/homebrew/bin"));

        Assert.Equal(Join("/opt/homebrew/bin", "/usr/bin", "/usr/local/bin", "/bin"), merged);
    }

    [Fact]
    public void WithoutAnAnswerFromTheShellTheFallbacksGoInFrontOfTheCurrentPath()
    {
        string merged = DesktopEnvironment.MergePath(null, new[] { "/opt/homebrew/bin" }, Join("/usr/bin", "/bin"));

        Assert.Equal(Join("/opt/homebrew/bin", "/usr/bin", "/bin"), merged);
    }

    [SkippableFact]
    public void OnAMacTheLoginShellsPathIsImportedAndASecondImportChangesNothing()
    {
        Skip.IfNot(OperatingSystem.IsMacOS(), "The login-shell import applies to macOS and Mac Catalyst only.");
        string? saved = Environment.GetEnvironmentVariable("PATH");
        try
        {
            // What a Finder launch starts with.
            Environment.SetEnvironmentVariable("PATH", Join("/usr/bin", "/bin", "/usr/sbin", "/sbin"));

            string? imported = DesktopEnvironment.ImportLoginShellPath();

            string now = Environment.GetEnvironmentVariable("PATH")!;
            // Whatever the user's profile adds, the system directories survive and come once.
            foreach (string system in new[] { "/usr/bin", "/bin", "/usr/sbin", "/sbin" })
                Assert.Single(now.Split(Path.PathSeparator), e => e == system);
            if (imported is not null)
                Assert.Equal(imported, now);

            Assert.Null(DesktopEnvironment.ImportLoginShellPath());
        }
        finally
        {
            Environment.SetEnvironmentVariable("PATH", saved);
        }
    }

    [SkippableFact]
    public void OffTheMacTheImportLeavesPathAlone()
    {
        Skip.If(OperatingSystem.IsMacOS(), "The import runs on macOS.");
        string? saved = Environment.GetEnvironmentVariable("PATH");
        Assert.Null(DesktopEnvironment.ImportLoginShellPath());
        Assert.Equal(saved, Environment.GetEnvironmentVariable("PATH"));
    }
}
