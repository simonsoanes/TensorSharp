// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Diagnostics;
using TensorAgent.Core.JavaScript;
using TensorAgent.Core.Sandbox;

namespace TensorAgent.Tests;

/// <summary>
/// These run for real. JavaScriptCore lives at the same absolute path on macOS as
/// on iOS, so every one of these tests loads the same framework the phone loads
/// and executes the same JavaScript — no mock engine, no skip when the platform
/// looks wrong. If JavaScriptCore is missing, the first assertion says so instead
/// of the suite quietly passing.
/// </summary>
public sealed class JavaScriptCoreEngineTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "tensoragent-js-" + Guid.NewGuid().ToString("N"));
    private readonly string _work;
    private readonly string _temp;
    private readonly string _outside;
    private readonly JavaScriptCoreEngine _engine = new();

    public JavaScriptCoreEngineTests()
    {
        _work = Path.Combine(_root, "work");
        _temp = Path.Combine(_root, "tmp");
        _outside = Path.Combine(_root, "outside");
        Directory.CreateDirectory(_work);
        Directory.CreateDirectory(_temp);
        Directory.CreateDirectory(_outside);
    }

    public void Dispose()
    {
        try { Directory.Delete(_root, true); } catch { }
    }

    private ExecutionPolicy Policy(bool network = false, bool scripts = true, int maxOutput = 32 * 1024) => new(
        AllowScripts: scripts,
        AllowNetwork: network,
        WorkRoot: _work,
        ReadableRoots: Array.Empty<string>(),
        TempRoot: _temp)
    {
        DefaultTimeout = TimeSpan.FromSeconds(20),
        MaxOutputBytes = maxOutput,
    };

    private InterpreterContext Context(ExecutionPolicy? policy = null, IReadOnlyDictionary<string, string>? environment = null,
        TimeSpan? timeout = null, string? standardInput = null)
        => new(_work, environment ?? new Dictionary<string, string> { ["HOME"] = _work }, policy ?? Policy())
        {
            Timeout = timeout,
            StandardInput = standardInput,
        };

    private ExecutionResult Eval(string code, InterpreterContext? context = null)
        => _engine.RunCodeAsync(code, Array.Empty<string>(), context ?? Context(), CancellationToken.None)
            .GetAwaiter().GetResult();

    private ExecutionResult RunFile(string relativePath, IReadOnlyList<string>? arguments = null, InterpreterContext? context = null)
    {
        string path = Path.Combine(_work, relativePath);
        var argv = new List<string> { relativePath };
        if (arguments is not null)
            argv.AddRange(arguments);
        return _engine.RunScriptAsync(path, argv, context ?? Context(), CancellationToken.None).GetAwaiter().GetResult();
    }

    private string Write(string relativePath, string contents)
    {
        string path = Path.Combine(_work, relativePath);
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllText(path, contents);
        return path;
    }

    private string Stdout(string code, InterpreterContext? context = null)
    {
        ExecutionResult result = Eval(code, context);
        Assert.True(result.ExitCode == 0, $"exit {result.ExitCode}: {result.Stderr}");
        return result.Stdout;
    }

    // ---- the engine itself -------------------------------------------------------------

    [AppleJavaScriptFact]
    public void TheFrameworkIsActuallyPresent()
    {
        Assert.True(_engine.IsAvailable, _engine.UnavailableReason);
        Assert.Null(_engine.UnavailableReason);
    }

    [AppleJavaScriptFact]
    public void EvaluatesAnExpressionAndExitsZero()
    {
        ExecutionResult result = Eval("const x = 6 * 7; console.log(x);");
        Assert.Equal(0, result.ExitCode);
        Assert.False(result.TimedOut);
        Assert.True(result.Ok);
        Assert.Equal("42\n", result.Stdout);
        Assert.Equal(string.Empty, result.Stderr);
        Assert.Equal(_work, result.WorkingDirectory);
    }

    [AppleJavaScriptFact]
    public void RunsAScriptFromAFile()
    {
        Write("main.js", "console.log('from a file');\nconsole.log(__filename.endsWith('main.js'));\n");
        ExecutionResult result = RunFile("main.js");
        Assert.Equal(0, result.ExitCode);
        Assert.Equal("from a file\ntrue\n", result.Stdout);
    }

    [AppleJavaScriptFact]
    public void ScriptsCanBeDisabledByPolicy()
    {
        ExecutionResult result = Eval("console.log(1)", Context(Policy(scripts: false)));
        Assert.Equal(126, result.ExitCode);
        Assert.Contains(ExecutionPolicy.ScriptsDisabledMessage, result.Stderr, StringComparison.Ordinal);
    }

    // ---- console -----------------------------------------------------------------------

    [AppleJavaScriptFact]
    public void ConsoleFormatsTheCommonCasesLikeNode()
    {
        Assert.Equal("plain string\n", Stdout("console.log('plain string')"));
        Assert.Equal("1 2.5 -0.5\n", Stdout("console.log(1, 2.5, -0.5)"));
        Assert.Equal("true false\n", Stdout("console.log(true, false)"));
        Assert.Equal("undefined null\n", Stdout("console.log(undefined, null)"));
        Assert.Equal("{\"a\":1,\"b\":[1,2]}\n", Stdout("console.log({a:1,b:[1,2]})"));
        Assert.Equal("[1,\"two\"]\n", Stdout("console.log([1,'two'])"));
        Assert.Equal("[Function: named]\n", Stdout("function named(){}; console.log(named)"));
        Assert.Equal("a b\n", Stdout("console.log('a', 'b')"));
    }

    [AppleJavaScriptFact]
    public void ConsoleErrorAndWarnGoToStandardError()
    {
        ExecutionResult result = Eval("console.log('out'); console.error('bad'); console.warn('careful');");
        Assert.Equal(0, result.ExitCode);
        Assert.Equal("out\n", result.Stdout);
        Assert.Equal("bad\ncareful\n", result.Stderr);
    }

    [AppleJavaScriptFact]
    public void OutputIsStreamedLineByLineToTheTap()
    {
        var lines = new List<string>();
        var context = Context() with { OnStdoutLine = lines.Add };
        Eval("console.log('one'); console.log('two');", context);
        Assert.Equal(new[] { "one", "two" }, lines);
    }

    // ---- errors ------------------------------------------------------------------------

    [AppleJavaScriptFact]
    public void AnUncaughtThrowExitsNonZeroWithANodeShapedStack()
    {
        Write("boom.js", "function inner() { throw new TypeError('exploded'); }\nfunction outer() { inner(); }\nouter();\n");
        ExecutionResult result = RunFile("boom.js");

        Assert.NotEqual(0, result.ExitCode);
        Assert.False(result.TimedOut);
        Assert.Contains("TypeError: exploded", result.Stderr, StringComparison.Ordinal);
        Assert.Contains("    at inner (", result.Stderr, StringComparison.Ordinal);
        Assert.Contains("    at outer (", result.Stderr, StringComparison.Ordinal);
        Assert.Contains("boom.js:1", result.Stderr, StringComparison.Ordinal);
    }

    [AppleJavaScriptFact]
    public void ThrowingANonErrorStillReports()
    {
        ExecutionResult result = Eval("throw 'just a string';");
        Assert.Equal(1, result.ExitCode);
        Assert.Contains("just a string", result.Stderr, StringComparison.Ordinal);
    }

    [AppleJavaScriptFact]
    public void AnErrorInATimerEndsTheRun()
    {
        ExecutionResult result = Eval("setTimeout(() => { throw new Error('late'); }, 1);");
        Assert.Equal(1, result.ExitCode);
        Assert.Contains("Error: late", result.Stderr, StringComparison.Ordinal);
    }

    // ---- process -----------------------------------------------------------------------

    [AppleJavaScriptFact]
    public void ProcessArgvNamesNodeThenTheScriptThenTheArguments()
    {
        Write("args.js", "console.log(JSON.stringify(process.argv));");
        ExecutionResult result = RunFile("args.js", new[] { "alpha", "beta" });
        Assert.Equal(0, result.ExitCode);

        string[] argv = System.Text.Json.JsonSerializer.Deserialize<string[]>(result.Stdout)!;
        Assert.Equal("node", argv[0]);
        Assert.EndsWith("args.js", argv[1], StringComparison.Ordinal);
        Assert.True(Path.IsPathRooted(argv[1]));
        Assert.Equal(new[] { "alpha", "beta" }, argv[2..]);
    }

    [AppleJavaScriptFact]
    public void ProcessEnvIsReadableAndComesBackOut()
    {
        var environment = new Dictionary<string, string> { ["GREETING"] = "hello", ["HOME"] = _work };
        ExecutionResult result = Eval(
            "console.log(process.env.GREETING); process.env.ADDED = 'yes';",
            Context(environment: environment));

        Assert.Equal(0, result.ExitCode);
        Assert.Equal("hello\n", result.Stdout);
        Assert.Equal("yes", result.Environment["ADDED"]);
        Assert.Equal("hello", result.Environment["GREETING"]);
    }

    [AppleJavaScriptFact]
    public void ProcessExitStopsTheScriptAndSetsTheStatus()
    {
        ExecutionResult result = Eval("console.log('before'); process.exit(3); console.log('after');");
        Assert.Equal(3, result.ExitCode);
        Assert.False(result.TimedOut);
        Assert.Equal("before\n", result.Stdout);
        Assert.DoesNotContain("after", result.Stdout, StringComparison.Ordinal);
    }

    [AppleJavaScriptFact]
    public void ProcessExitFromATimerStopsTheLoop()
    {
        ExecutionResult result = Eval("setTimeout(() => process.exit(7), 1); setInterval(() => console.log('tick'), 5);");
        Assert.Equal(7, result.ExitCode);
        Assert.False(result.TimedOut);
    }

    [AppleJavaScriptFact]
    public void ProcessBasicsAreThere()
    {
        Assert.Equal("ios\n", Stdout("console.log(process.platform)"));
        Assert.Equal(_work + "\n", Stdout("console.log(process.cwd())"));
        Assert.Equal("written", Stdout("process.stdout.write('written')"));
        ExecutionResult result = Eval("process.stderr.write('to stderr')");
        Assert.Equal("to stderr", result.Stderr);
    }

    [AppleJavaScriptFact]
    public void StandardInputIsReadable()
    {
        ExecutionResult result = Eval("console.log(process.stdin.read().trim().toUpperCase())",
            Context(standardInput: "piped in\n"));
        Assert.Equal(0, result.ExitCode);
        Assert.Equal("PIPED IN\n", result.Stdout);
    }

    // ---- fs ----------------------------------------------------------------------------

    [AppleJavaScriptFact]
    public void FileWritesInsideTheWorkRootSucceed()
    {
        ExecutionResult result = Eval(
            "const fs = require('fs');" +
            "fs.writeFileSync('note.txt', 'hello');" +
            "fs.appendFileSync('note.txt', ' again');" +
            "console.log(fs.readFileSync('note.txt', 'utf8'));" +
            "console.log(fs.existsSync('note.txt'), fs.existsSync('missing.txt'));");

        Assert.Equal(0, result.ExitCode);
        Assert.Equal("hello again\ntrue false\n", result.Stdout);
        Assert.Equal("hello again", File.ReadAllText(Path.Combine(_work, "note.txt")));
    }

    [AppleJavaScriptFact]
    public void AWriteOutsideTheWorkRootIsRefusedWithThePolicysOwnWording()
    {
        string target = Path.Combine(_outside, "stolen.txt").Replace("\\", "/");
        ExecutionResult result = Eval(
            "const fs = require('fs');" +
            "try { fs.writeFileSync(" + Quote(target) + ", 'x'); console.log('WROTE'); }" +
            "catch (e) { console.log(e.code); console.log(e.message); }");

        Assert.Equal(0, result.ExitCode);
        Assert.DoesNotContain("WROTE", result.Stdout, StringComparison.Ordinal);
        Assert.Contains("EACCES", result.Stdout, StringComparison.Ordinal);
        // Exactly what ConfinedPaths says, so the model reads one sentence everywhere.
        Assert.Contains(new ConfinementException(target).Message, result.Stdout, StringComparison.Ordinal);
        Assert.False(File.Exists(Path.Combine(_outside, "stolen.txt")));
    }

    [AppleJavaScriptFact]
    public void AReadOutsideTheWorkRootIsRefused()
    {
        File.WriteAllText(Path.Combine(_outside, "secret.txt"), "top secret");
        string target = Path.Combine(_outside, "secret.txt").Replace("\\", "/");
        ExecutionResult result = Eval(
            "const fs = require('fs');" +
            "try { console.log(fs.readFileSync(" + Quote(target) + ", 'utf8')); }" +
            "catch (e) { console.log('refused:' + e.code); }");

        Assert.Equal(0, result.ExitCode);
        Assert.Equal("refused:EACCES\n", result.Stdout);
        Assert.DoesNotContain("top secret", result.Stdout, StringComparison.Ordinal);
    }

    [AppleJavaScriptFact]
    public void ASymlinkPointingOutOfTheWorkspaceIsRefused()
    {
        File.WriteAllText(Path.Combine(_outside, "secret.txt"), "top secret");
        string link = Path.Combine(_work, "escape.txt");
        File.CreateSymbolicLink(link, Path.Combine(_outside, "secret.txt"));
        Assert.True(File.Exists(link), "the symlink was not created");

        ExecutionResult result = Eval(
            "const fs = require('fs');" +
            "try { console.log('LEAKED:' + fs.readFileSync('escape.txt', 'utf8')); }" +
            "catch (e) { console.log('refused:' + e.code); }" +
            "console.log('exists=' + fs.existsSync('escape.txt'));");

        Assert.Equal(0, result.ExitCode);
        Assert.DoesNotContain("top secret", result.Stdout, StringComparison.Ordinal);
        Assert.Contains("refused:EACCES", result.Stdout, StringComparison.Ordinal);
        // existsSync answers false rather than throwing, which is Node's contract.
        Assert.Contains("exists=false", result.Stdout, StringComparison.Ordinal);
    }

    [AppleJavaScriptFact]
    public void ASymlinkedDirectoryCannotBeWrittenThrough()
    {
        string link = Path.Combine(_work, "out");
        Directory.CreateSymbolicLink(link, _outside);
        ExecutionResult result = Eval(
            "const fs = require('fs');" +
            "try { fs.writeFileSync('out/planted.txt', 'x'); console.log('WROTE'); }" +
            "catch (e) { console.log('refused:' + e.code); }");

        Assert.Equal(0, result.ExitCode);
        Assert.Equal("refused:EACCES\n", result.Stdout);
        Assert.False(File.Exists(Path.Combine(_outside, "planted.txt")));
    }

    [AppleJavaScriptFact]
    public void DirectoriesAndStatsAndRemoval()
    {
        ExecutionResult result = Eval(
            "const fs = require('fs');" +
            "fs.mkdirSync('nested/deep', { recursive: true });" +
            "fs.writeFileSync('nested/deep/a.txt', 'abcd');" +
            "fs.writeFileSync('nested/deep/b.txt', 'ef');" +
            "console.log(JSON.stringify(fs.readdirSync('nested/deep')));" +
            "const st = fs.statSync('nested/deep/a.txt');" +
            "console.log(st.size, st.isFile(), st.isDirectory());" +
            "console.log(fs.statSync('nested').isDirectory());" +
            "fs.rmSync('nested', { recursive: true });" +
            "console.log(fs.existsSync('nested'));");

        Assert.Equal(0, result.ExitCode);
        Assert.Equal("[\"a.txt\",\"b.txt\"]\n4 true false\ntrue\nfalse\n", result.Stdout);
    }

    [AppleJavaScriptFact]
    public void MissingFilesRaiseEnoent()
    {
        ExecutionResult result = Eval(
            "const fs = require('fs');" +
            "try { fs.readFileSync('nope.txt', 'utf8'); } catch (e) { console.log(e.code, e.message); }");
        Assert.Equal(0, result.ExitCode);
        Assert.Contains("ENOENT", result.Stdout, StringComparison.Ordinal);
        Assert.Contains("no such file or directory", result.Stdout, StringComparison.Ordinal);
    }

    [AppleJavaScriptFact]
    public void FsPromisesWork()
    {
        ExecutionResult result = Eval(
            "const fs = require('fs');" +
            "(async () => {" +
            "  await fs.promises.writeFile('async.txt', 'via promises');" +
            "  console.log(await fs.promises.readFile('async.txt', 'utf8'));" +
            "  try { await fs.promises.readFile('gone.txt', 'utf8'); }" +
            "  catch (e) { console.log('rejected:' + e.code); }" +
            "})();");

        Assert.Equal(0, result.ExitCode);
        Assert.Equal("via promises\nrejected:ENOENT\n", result.Stdout);
    }

    [AppleJavaScriptFact]
    public void ReadingWithoutAnEncodingGivesABuffer()
    {
        ExecutionResult result = Eval(
            "const fs = require('fs');" +
            "fs.writeFileSync('bytes.bin', 'hi');" +
            "const buffer = fs.readFileSync('bytes.bin');" +
            "console.log(Buffer.isBuffer(buffer), buffer.length, buffer.toString('hex'), buffer.toString('base64'));");

        Assert.Equal(0, result.ExitCode);
        Assert.Equal("true 2 6869 aGk=\n", result.Stdout);
    }

    // ---- path and os -------------------------------------------------------------------

    [AppleJavaScriptFact]
    public void PathBehavesLikeNodesPosixPath()
    {
        Assert.Equal("a/b/c\n", Stdout("const p=require('path'); console.log(p.join('a','b','c'))"));
        Assert.Equal("a/c\n", Stdout("const p=require('path'); console.log(p.join('a','b','..','c'))"));
        Assert.Equal(".\n", Stdout("const p=require('path'); console.log(p.join())"));
        Assert.Equal("/x/y\n", Stdout("const p=require('path'); console.log(p.resolve('/x','y'))"));
        Assert.Equal("/a/b\n", Stdout("const p=require('path'); console.log(p.dirname('/a/b/c.txt'))"));
        Assert.Equal(".\n", Stdout("const p=require('path'); console.log(p.dirname('c.txt'))"));
        Assert.Equal("/\n", Stdout("const p=require('path'); console.log(p.dirname('/c.txt'))"));
        Assert.Equal("c.txt\n", Stdout("const p=require('path'); console.log(p.basename('/a/b/c.txt'))"));
        Assert.Equal("c\n", Stdout("const p=require('path'); console.log(p.basename('/a/b/c.txt','.txt'))"));
        Assert.Equal(".txt\n", Stdout("const p=require('path'); console.log(p.extname('/a/b/c.txt'))"));
        Assert.Equal("\n", Stdout("const p=require('path'); console.log(p.extname('.gitignore'))"));
        Assert.Equal("../c/d\n", Stdout("const p=require('path'); console.log(p.relative('/a/b','/a/c/d'))"));
        Assert.Equal("/\n", Stdout("const p=require('path'); console.log(p.sep)"));
        Assert.Equal("true false\n", Stdout("const p=require('path'); console.log(p.isAbsolute('/a'), p.isAbsolute('a'))"));
        // path.resolve with no absolute part falls back to the working directory.
        Assert.Equal(_work + "/here\n", Stdout("const p=require('path'); console.log(p.resolve('here'))"));
    }

    [AppleJavaScriptFact]
    public void OsReportsTheSandboxsOwnDirectories()
    {
        Assert.Equal("ios\n", Stdout("console.log(require('os').platform())"));
        Assert.Equal(_temp + "\n", Stdout("console.log(require('os').tmpdir())"));
        Assert.Equal(_work + "\n", Stdout("console.log(require('os').homedir())"));
        Assert.Equal("1\n", Stdout("console.log(require('os').EOL.length)"));
    }

    // ---- require -----------------------------------------------------------------------

    [AppleJavaScriptFact]
    public void RequireLoadsARelativeFileInACommonJsWrapper()
    {
        Write("lib/greet.js",
            "var hidden = 'not a global';\n" +
            "module.exports.greet = function (name) { return 'hello ' + name; };\n" +
            "module.exports.where = { file: __filename, dir: __dirname };\n");
        Write("main.js",
            "const lib = require('./lib/greet.js');\n" +
            "console.log(lib.greet('world'));\n" +
            "console.log(lib.where.file.endsWith('lib/greet.js'), lib.where.dir.endsWith('/lib'));\n" +
            "console.log(typeof hidden);\n");

        ExecutionResult result = RunFile("main.js");
        Assert.Equal(0, result.ExitCode);
        Assert.Equal("hello world\ntrue true\nundefined\n", result.Stdout);
    }

    [AppleJavaScriptFact]
    public void RequireResolvesWithoutAnExtensionAndCachesTheModule()
    {
        Write("counter.js", "let n = 0;\nmodule.exports = function () { return ++n; };\n");
        Write("main.js",
            "const first = require('./counter');\n" +
            "const second = require('./counter.js');\n" +
            "console.log(first === second, first(), second(), first());\n");

        ExecutionResult result = RunFile("main.js");
        Assert.Equal(0, result.ExitCode);
        // One module object, one closure, one counter — that is what caching means.
        Assert.Equal("true 1 2 3\n", result.Stdout);
    }

    [AppleJavaScriptFact]
    public void RequireLoadsAnIndexFileAndJson()
    {
        Write("pkg/index.js", "module.exports = { name: 'pkg' };\n");
        Write("data.json", "{\"answer\": 42}\n");
        Write("main.js", "console.log(require('./pkg').name, require('./data.json').answer);\n");

        ExecutionResult result = RunFile("main.js");
        Assert.Equal(0, result.ExitCode);
        Assert.Equal("pkg 42\n", result.Stdout);
    }

    [AppleJavaScriptFact]
    public void RequireOfAnNpmModuleSaysThisHostHasNoNpm()
    {
        ExecutionResult result = Eval("require('http');");
        Assert.Equal(1, result.ExitCode);
        Assert.Contains("Cannot find module 'http'", result.Stderr, StringComparison.Ordinal);
        Assert.Contains("no npm", result.Stderr, StringComparison.Ordinal);
        Assert.Contains("fs, path, os", result.Stderr, StringComparison.Ordinal);
    }

    [AppleJavaScriptFact]
    public void RequireOfAnUnknownPackageIsAlsoNamed()
    {
        ExecutionResult result = Eval("try { require('lodash'); } catch (e) { console.log(e.code, e.message); }");
        Assert.Equal(0, result.ExitCode);
        Assert.Contains("MODULE_NOT_FOUND", result.Stdout, StringComparison.Ordinal);
        Assert.Contains("lodash", result.Stdout, StringComparison.Ordinal);
    }

    [AppleJavaScriptFact]
    public void RequireOfAFileOutsideTheWorkspaceIsRefused()
    {
        File.WriteAllText(Path.Combine(_outside, "evil.js"), "module.exports = 'pwned';\n");
        Write("main.js",
            "try { console.log(require('../outside/evil.js')); }\n" +
            "catch (e) { console.log('refused:' + e.code); }\n");

        ExecutionResult result = RunFile("main.js");
        Assert.Equal(0, result.ExitCode);
        Assert.Equal("refused:EACCES\n", result.Stdout);
    }

    [AppleJavaScriptFact]
    public void AModuleThatThrowsPropagatesItsOwnError()
    {
        Write("bad.js", "throw new RangeError('module blew up');\n");
        Write("main.js", "try { require('./bad'); } catch (e) { console.log(e.name, e.message); }\n");

        ExecutionResult result = RunFile("main.js");
        Assert.Equal(0, result.ExitCode);
        Assert.Equal("RangeError module blew up\n", result.Stdout);
    }

    // ---- timers, microtasks, promises --------------------------------------------------

    [AppleJavaScriptFact]
    public void TimersRunInDeadlineOrderAfterTheSynchronousScript()
    {
        ExecutionResult result = Eval(
            "const order = [];" +
            "setTimeout(() => order.push('t20'), 20);" +
            "setTimeout(() => order.push('t1'), 1);" +
            "order.push('sync');" +
            "setTimeout(() => console.log(order.join(',')), 40);");

        Assert.Equal(0, result.ExitCode);
        Assert.Equal("sync,t1,t20\n", result.Stdout);
    }

    [AppleJavaScriptFact]
    public void MicrotasksDrainBeforeTimers()
    {
        ExecutionResult result = Eval(
            "const order = [];" +
            "setTimeout(() => { order.push('timer'); console.log(order.join(',')); }, 5);" +
            "Promise.resolve().then(() => order.push('promise'));" +
            "queueMicrotask(() => order.push('micro'));" +
            "order.push('sync');");

        Assert.Equal(0, result.ExitCode);
        Assert.Equal("sync,promise,micro,timer\n", result.Stdout);
    }

    [AppleJavaScriptFact]
    public void AsyncAwaitAcrossATimerCompletes()
    {
        ExecutionResult result = Eval(
            "const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));" +
            "(async () => { await sleep(5); console.log('one'); await sleep(5); console.log('two'); })();");

        Assert.Equal(0, result.ExitCode);
        Assert.Equal("one\ntwo\n", result.Stdout);
    }

    [AppleJavaScriptFact]
    public void ClearTimeoutCancelsAndClearIntervalStops()
    {
        ExecutionResult result = Eval(
            "const cancelled = setTimeout(() => console.log('SHOULD NOT RUN'), 5);" +
            "clearTimeout(cancelled);" +
            "let ticks = 0;" +
            "const interval = setInterval(() => { if (++ticks === 3) { clearInterval(interval); console.log('ticks=' + ticks); } }, 2);");

        Assert.Equal(0, result.ExitCode);
        Assert.Equal("ticks=3\n", result.Stdout);
        Assert.DoesNotContain("SHOULD NOT RUN", result.Stdout, StringComparison.Ordinal);
    }

    [AppleJavaScriptFact]
    public void APromiseThatNeverSettlesEndsAtTheDeadlineInsteadOfHanging()
    {
        var clock = Stopwatch.StartNew();
        ExecutionResult result = Eval(
            "new Promise(() => {}).then(() => console.log('never'));" +
            "setInterval(() => {}, 10);",
            Context(timeout: TimeSpan.FromSeconds(1)));
        clock.Stop();

        Assert.True(result.TimedOut, "the run should have reported a timeout");
        Assert.Equal(ExecutionResult.TimeoutExitCode, result.ExitCode);
        Assert.True(clock.Elapsed < TimeSpan.FromSeconds(6), $"took {clock.Elapsed}");
        Assert.Contains("deadline", result.Stderr, StringComparison.Ordinal);
    }

    // ---- the timeout, and whether a runaway loop is really stoppable --------------------

    [AppleJavaScriptFact]
    public void TheExecutionTimeLimitSymbolResolves()
    {
        // If this ever goes false on a future OS, the next test documents what the
        // engine falls back to; it must never fail silently.
        Assert.True(_engine.CanInterruptRunawayScripts,
            "JSContextGroupSetExecutionTimeLimit did not resolve; the engine cannot stop a runaway script on this host");
    }

    /// <summary>
    /// The one that matters. An empty <c>while (true) {}</c> allocates nothing,
    /// calls nothing and never leaves optimized code, so nothing but a real
    /// interrupt can end it — which is exactly why it is the loop worth testing.
    /// </summary>
    [AppleJavaScriptTheory]
    [InlineData("while (true) {}")]
    [InlineData("let n = 0; while (true) { n++; }")]
    [InlineData("while (true) { Math.sqrt(2); }")]
    [InlineData("const a = []; while (true) { a.length = 0; a.push(1); }")]
    public void ARunawayLoopIsActuallyStopped(string code)
    {
        var clock = Stopwatch.StartNew();
        ExecutionResult result = Eval(code, Context(timeout: TimeSpan.FromSeconds(1)));
        clock.Stop();

        Assert.True(result.TimedOut, "the run should have reported a timeout: " + result.Stderr);
        Assert.Equal(ExecutionResult.TimeoutExitCode, result.ExitCode);
        // Interrupted ON the deadline, not merely abandoned two seconds past it.
        Assert.True(clock.Elapsed < TimeSpan.FromSeconds(2),
            $"the loop was not interrupted at the deadline: {clock.Elapsed} / {result.Stderr}");
        Assert.Contains("stopped by JavaScriptCore's execution watchdog", result.Stderr, StringComparison.Ordinal);
        Assert.DoesNotContain("cannot preempt", result.Stderr, StringComparison.Ordinal);
    }

    [AppleJavaScriptFact]
    public void ARunawayLoopInsideATimerIsAlsoStopped()
    {
        var clock = Stopwatch.StartNew();
        ExecutionResult result = Eval("setTimeout(() => { while (true) {} }, 1);",
            Context(timeout: TimeSpan.FromSeconds(1)));
        clock.Stop();

        Assert.True(result.TimedOut, result.Stderr);
        Assert.Equal(ExecutionResult.TimeoutExitCode, result.ExitCode);
        Assert.True(clock.Elapsed < TimeSpan.FromSeconds(2), $"took {clock.Elapsed}");
    }

    [AppleJavaScriptFact]
    public void ARunawayLoopInsideARequiredModuleIsStopped()
    {
        Write("spin.js", "while (true) {}\n");
        Write("main.js", "require('./spin');\n");
        var clock = Stopwatch.StartNew();
        ExecutionResult result = RunFile("main.js", context: Context(timeout: TimeSpan.FromSeconds(1)));
        clock.Stop();

        Assert.True(result.TimedOut, result.Stderr);
        Assert.True(clock.Elapsed < TimeSpan.FromSeconds(2), $"took {clock.Elapsed}");
    }

    // ---- network -----------------------------------------------------------------------

    [AppleJavaScriptFact]
    public void FetchIsDefinedAndThrowsThePolicysNetworkMessageWhenTheNetworkIsOff()
    {
        ExecutionResult result = Eval(
            "console.log(typeof fetch);" +
            "try { fetch('https://example.com'); } catch (e) { console.log(e.message); }");

        Assert.Equal(0, result.ExitCode);
        Assert.Equal("function\n" + ExecutionPolicy.NetworkDisabledMessage + "\n", result.Stdout);
    }

    [AppleJavaScriptFact]
    public void AnAwaitedFetchRejectsWithTheSameSentence()
    {
        ExecutionResult result = Eval(
            "(async () => { try { await fetch('https://example.com'); } catch (e) { console.log('caught: ' + e.message); } })();");
        Assert.Equal(0, result.ExitCode);
        Assert.Equal("caught: " + ExecutionPolicy.NetworkDisabledMessage + "\n", result.Stdout);
    }

    [AppleJavaScriptFact]
    public void FetchExistsWhenTheNetworkIsOnButStillHonoursTheHostAllowList()
    {
        var policy = Policy(network: true) with { NetworkHosts = new[] { "example.com" } };
        ExecutionResult result = Eval(
            "(async () => { try { await fetch('https://not-allowed.invalid/x'); } catch (e) { console.log(e.message); } })();",
            Context(policy));

        Assert.Equal(0, result.ExitCode);
        Assert.Contains("not-allowed.invalid is not in this session's allowed hosts", result.Stdout, StringComparison.Ordinal);
    }

    // ---- text, buffers, encodings ------------------------------------------------------

    [AppleJavaScriptFact]
    public void BufferAndTextEncoderCoverWhatScriptsUse()
    {
        Assert.Equal("aGVsbG8=\n", Stdout("console.log(Buffer.from('hello').toString('base64'))"));
        Assert.Equal("hello\n", Stdout("console.log(Buffer.from('aGVsbG8=', 'base64').toString('utf8'))"));
        Assert.Equal("68656c6c6f\n", Stdout("console.log(Buffer.from('hello').toString('hex'))"));
        Assert.Equal("true\n", Stdout("console.log(Buffer.from('x') instanceof Uint8Array)"));
        Assert.Equal("ab\n", Stdout("console.log(Buffer.concat([Buffer.from('a'), Buffer.from('b')]).toString())"));
        Assert.Equal("3\n", Stdout("console.log(new TextEncoder().encode('a\\u00e9').length)"));
        Assert.Equal("héllo\n", Stdout("console.log(new TextDecoder().decode(new TextEncoder().encode('héllo')))"));
    }

    // ---- syntax check ------------------------------------------------------------------

    [AppleJavaScriptFact]
    public async Task SyntaxCheckPassesOnValidSource()
    {
        string path = Write("ok.js", "const a = 1;\nmodule.exports = a;\n");
        SyntaxCheckResult result = await _engine.CheckSyntaxAsync(path, CancellationToken.None);
        Assert.True(result.Ok, result.Message);
        Assert.Null(result.Message);
    }

    [AppleJavaScriptFact]
    public async Task SyntaxCheckAcceptsATopLevelReturn()
    {
        // Legal in a CommonJS module, so `node --check` accepts it and so does this.
        string path = Write("early.js", "if (process.env.SKIP) { return; }\nmodule.exports = 1;\n");
        SyntaxCheckResult result = await _engine.CheckSyntaxAsync(path, CancellationToken.None);
        Assert.True(result.Ok, result.Message);
    }

    [AppleJavaScriptFact]
    public async Task SyntaxCheckReportsPathAndLine()
    {
        string path = Write("broken.js", "const a = 1;\nconst b = ;\nconst c = 3;\n");
        SyntaxCheckResult result = await _engine.CheckSyntaxAsync(path, CancellationToken.None);

        Assert.False(result.Ok);
        Assert.NotNull(result.Message);
        Assert.StartsWith(path + ":2: ", result.Message, StringComparison.Ordinal);
    }

    // ---- output bounds -----------------------------------------------------------------

    [AppleJavaScriptFact]
    public void OutputIsCappedAtThePolicysLimit()
    {
        ExecutionResult result = Eval(
            "for (let i = 0; i < 4000; i++) console.log('line ' + i + ' ' + 'x'.repeat(60));",
            Context(Policy(maxOutput: 8 * 1024)));

        Assert.Equal(0, result.ExitCode);
        Assert.True(result.Stdout.Length < 32 * 1024, $"output was {result.Stdout.Length} bytes");
        Assert.Contains("were dropped from the middle", result.Stdout, StringComparison.Ordinal);
        // Both ends survive: the first line and the last are what answer "did it work".
        Assert.Contains("line 0 ", result.Stdout, StringComparison.Ordinal);
        Assert.Contains("line 3999 ", result.Stdout, StringComparison.Ordinal);
    }

    [AppleJavaScriptFact]
    public void ACircularObjectPrintsSomethingRatherThanCrashing()
    {
        // JSON.stringify refuses a cycle; console.log must still come back.
        ExecutionResult result = Eval("const a = {}; a.self = a; console.log(a); console.log('after');");
        Assert.Equal(0, result.ExitCode);
        Assert.Contains("after", result.Stdout, StringComparison.Ordinal);
    }

    [AppleJavaScriptFact]
    public void AnErrorObjectPrintsItsNameMessageAndStack()
    {
        ExecutionResult result = Eval("function f(){ return new Error('inspect me'); } console.log(f());");
        Assert.Equal(0, result.ExitCode);
        Assert.StartsWith("Error: inspect me\n    at ", result.Stdout, StringComparison.Ordinal);
    }

    [AppleJavaScriptFact]
    public void CancellingTheRunThrows()
    {
        using var cancellation = new CancellationTokenSource();
        cancellation.CancelAfter(TimeSpan.FromMilliseconds(100));
        Assert.Throws<OperationCanceledException>(() =>
            _engine.RunCodeAsync("while (true) {}", Array.Empty<string>(),
                Context(timeout: TimeSpan.FromSeconds(30)), cancellation.Token).GetAwaiter().GetResult());
    }

    // ---- isolation ---------------------------------------------------------------------

    [AppleJavaScriptFact]
    public void EachRunGetsAFreshVirtualMachine()
    {
        Assert.Equal("undefined\n", Stdout("globalThis.leaked = 'x'; console.log(typeof globalThis.previous)"));
        Assert.Equal("undefined\n", Stdout("console.log(typeof globalThis.leaked)"));
    }

    private static string Quote(string value) => "'" + value.Replace("\\", "\\\\").Replace("'", "\\'") + "'";
}
