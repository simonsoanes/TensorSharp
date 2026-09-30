// Run: node --test eng/validation/tests/playwright_wrapper.test.mjs
// No browser, registry access, third-party packages or Bash are required.
// Fake npm/npx verifies the wrapper boundary, not npm's own command shim.
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { chmodSync, copyFileSync, existsSync, linkSync, mkdirSync, mkdtempSync, readFileSync, rmSync, symlinkSync, writeFileSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { afterEach, beforeEach, test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { buildCliArgs, createInvocation, resolveNpxCli } from '../../../TensorAgent/skills/playwright/scripts/playwright_cli.mjs';

const wrapper = fileURLToPath(new URL('../../../TensorAgent/skills/playwright/scripts/playwright_cli.mjs', import.meta.url));
const prefix = ['--yes', '--prefer-offline', '--package', '@playwright/cli@0.1.21', 'playwright-cli'];
let root;
beforeEach(() => { root = mkdtempSync(path.join(os.tmpdir(), 'playwright wrapper ')); });
afterEach(() => { rmSync(root, { recursive: true, force: true }); });

function args(...supplied) {
  const actual = buildCliArgs(supplied, { cwd: root, env: {} });
  assert.deepEqual(actual.slice(0, prefix.length), prefix);
  return actual.slice(prefix.length);
}

function generatedFile(actual) {
  const names = actual.filter(arg => arg.startsWith('--filename=')).map(arg => arg.slice('--filename='.length));
  assert.equal(names.length, 1);
  assert.match(names[0], /^output\/playwright\/snapshot-[\w-]+\.yml$/);
  assert.equal(readFileSync(path.join(root, names[0]), 'utf8'), '');
  return names[0];
}

function makeNpm(directory, source = '') {
  const script = path.join(directory, 'node_modules', 'npm', 'bin', 'npx-cli.js');
  mkdirSync(path.dirname(script), { recursive: true });
  writeFileSync(script, source);
  return script;
}

test('bare snapshot arguments direct output to distinct reserved files', () => {
  const first = args('snapshot');
  const second = args('snapshot');
  assert.equal(first.at(-1), 'snapshot');
  assert.notEqual(generatedFile(first), generatedFile(second));
});

test('snapshot selectors, spaces and preceding options retain exact boundaries', () => {
  for (const supplied of [
    ['--json', '--session', 'named session', 'snapshot', 'main > article', '--depth', '3', '--boxes'],
    ['--depth', '3', '--boxes', 'snapshot'],
    ['--raw', '--session=snapshot', 'snapshot'],
    ['--raw', 'false', '--json', 'true', 'snapshot'],
    ['--boxes', 'true', '--help', 'false', 'snapshot'],
    ['--config', 'config with spaces.json', 'snapshot'],
    ['-s', 'snapshot', 'snapshot'],
  ]) {
    const actual = args(...supplied);
    generatedFile(actual);
    assert.deepEqual(actual.slice(1), supplied);
  }
});

test('all explicit filename forms are preserved without artifacts', () => {
  for (const supplied of [
    ['snapshot', '--filename', 'chosen name.yml'], ['snapshot', '--filename=chosen name.yml'],
    ['snapshot', '--filename='], ['--filename', 'chosen.yml', 'snapshot'],
  ]) assert.deepEqual(args(...supplied), supplied);
  assert.equal(existsSync(path.join(root, 'output')), false);
});

test('help and version forms never create snapshots', () => {
  for (const supplied of [
    ['snapshot', '--help'], ['--help', 'snapshot'], ['snapshot', '-h'],
    ['--version', 'snapshot'], ['--help=true', 'snapshot'], ['--version', 'true', 'snapshot'],
    ['-v', 'snapshot'], ['--version=true', 'snapshot'],
  ]) assert.deepEqual(args(...supplied), supplied);
  assert.equal(existsSync(path.join(root, 'output')), false);
});

test('explicit false help and version flags retain bounded snapshots', () => {
  for (const supplied of [
    ['--help=false', 'snapshot'], ['--version', 'false', 'snapshot'],
    ['--version=false', 'snapshot'], ['--help', 'false', 'snapshot'],
  ]) {
    const actual = args(...supplied);
    generatedFile(actual);
    assert.deepEqual(actual.slice(1), supplied);
  }
});

test('snapshot as another command argument or session name is untouched', () => {
  for (const supplied of [
    ['fill', 'e3', 'snapshot'], ['open', 'snapshot'],
    ['--session', 'snapshot', 'goto', 'https://example.test'], ['-s=snapshot', 'close'],
  ]) assert.deepEqual(args(...supplied), supplied);
  assert.equal(existsSync(path.join(root, 'output')), false);
});

test('literal flags cannot suppress filename or environment session injection', () => {
  const supplied = ['snapshot', '--', '--filename=literal-target', '--session=literal-target'];
  const actual = buildCliArgs(supplied, { cwd: root, env: { PLAYWRIGHT_CLI_SESSION: 'environment session' } }).slice(prefix.length);
  assert.deepEqual(actual.slice(0, 2), ['--session', 'environment session']);
  assert.match(actual[2], /^--filename=output\/playwright\//);
  assert.deepEqual(actual.slice(3), supplied);
});

test('explicit session flags override environment defaults', () => {
  const env = { PLAYWRIGHT_CLI_SESSION: 'environment session' };
  const actual = buildCliArgs(['snapshot'], { cwd: root, env }).slice(prefix.length);
  assert.deepEqual(actual.slice(0, 2), ['--session', 'environment session']);
  generatedFile(actual);
  for (const flag of [['--session', 'supplied'], ['--session=supplied'], ['-s', 'supplied'], ['-s=supplied']]) {
    const supplied = [...flag, 'snapshot'];
    const forwarded = buildCliArgs(supplied, { cwd: root, env }).slice(prefix.length);
    generatedFile(forwarded);
    assert.deepEqual(forwarded.slice(1), supplied);
  }
});

test('ordinary open does not gain launch flags and manual handoff flags survive', () => {
  for (const supplied of [['open', 'https://example.test'],
    ['--session', 'account', 'open', 'https://example.test/login', '--headed', '--persistent']]) {
    assert.deepEqual(args(...supplied), supplied);
  }
});

test('native npm beside node takes precedence over PATH launchers', () => {
  const besideNode = path.join(root, 'native node');
  const other = path.join(root, 'another npm');
  const expected = makeNpm(besideNode);
  makeNpm(other);
  assert.equal(resolveNpxCli({ execPath: path.join(besideNode, 'node.exe'), env: { PATH: other } }), expected);
});

test('native npm resolves case-insensitive Windows Path entries with spaces', () => {
  const other = path.join(root, 'native npm');
  const expected = makeNpm(other);
  assert.equal(resolveNpxCli({ execPath: path.join(root, 'node.exe'),
    env: { Path: `${path.join(root, 'missing')};"${other}"` } }), expected);
});

test('missing native npm fails clearly without executing WSL or creating snapshots', () => {
  writeFileSync(path.join(root, 'npx.cmd'), '@echo fake WSL shim');
  assert.throws(() => createInvocation(['snapshot'], {
    platform: 'win32', execPath: path.join(root, 'node.exe'), env: { PATH: root }, cwd: root,
  }), /Native Windows npm was not found.*WSL/);
  assert.equal(existsSync(path.join(root, 'output')), false);
});

test('Windows invocation executes native Node plus JS entry with no shell command', () => {
  const script = makeNpm(root);
  const supplied = ['fill', 'e3', '你好 & whoami | "quoted" %PATH% $HOME `literal`'];
  const invocation = createInvocation(supplied, {
    platform: 'win32', execPath: path.join(root, 'node.exe'), env: {}, cwd: root,
  });
  assert.equal(invocation.command, path.join(root, 'node.exe'));
  assert.deepEqual(invocation.args, [script, ...prefix, ...supplied]);
});

function subprocessFixture() {
  // Windows resolves npm beside node.exe, so the fixture needs its own node.exe next
  // to the fake npm. POSIX runs npx from PATH and needs no copy - and a copied Node
  // cannot start when the binary loads a shared libnode through @rpath (Homebrew).
  const executable = process.platform === 'win32' ? path.join(root, 'node.exe') : process.execPath;
  if (process.platform === 'win32') {
    try { linkSync(process.execPath, executable); }
    catch { copyFileSync(process.execPath, executable); }
  }
  const source = `
    const child = require('node:child_process').spawnSync('node', ['-e', 'process.stdout.write("native child")'], { encoding: 'utf8' });
    if (child.status !== 0 || child.stdout !== 'native child') throw new Error('npm descendant could not find native Node');
    console.log(JSON.stringify(process.argv.slice(2)));
    process.exitCode = Number(process.env.FAKE_NPX_EXIT || 0);
  `;
  makeNpm(root, source);
  if (process.platform !== 'win32') {
    const npx = path.join(root, 'npx');
    writeFileSync(npx, '#!/usr/bin/env node\n' + source);
    chmodSync(npx, 0o755);
  }
  const env = { ...process.env, FAKE_NPX_EXIT: '23' };
  for (const key of Object.keys(env)) if (key.toLowerCase() === 'path') delete env[key];
  // Windows must support Node invoked by absolute path without Node on PATH.
  // Unix's npx executable is intentionally resolved through its normal PATH.
  env.PATH = process.platform === 'win32' ? '' : root + path.delimiter + (process.env.PATH || '');
  delete env.PLAYWRIGHT_CLI_SESSION;
  return { executable, env };
}

test('real subprocess forwards Unicode/metacharacters to fake npm and propagates its exit', () => {
  const { executable, env } = subprocessFixture();
  const supplied = ['fill', 'e3', '你好 & echo bad | "quotes" %PATH% $HOME `literal` ; (value) \\'];
  const result = spawnSync(executable, [wrapper, ...supplied], { cwd: root, env, encoding: 'utf8', timeout: 10_000 });
  assert.ifError(result.error);
  assert.equal(result.stderr, '');
  assert.equal(result.status, 23);
  assert.deepEqual(JSON.parse(result.stdout), [...prefix, ...supplied]);
});

test('a skill installed through a directory symlink or junction still executes', () => {
  const { executable, env } = subprocessFixture();
  const linked = path.join(root, 'linked skill');
  symlinkSync(path.dirname(wrapper), linked, process.platform === 'win32' ? 'junction' : 'dir');
  const result = spawnSync(executable, [path.join(linked, path.basename(wrapper)), '--version'], {
    cwd: root, env, encoding: 'utf8', timeout: 10_000,
  });
  assert.ifError(result.error);
  assert.equal(result.stderr, '');
  assert.equal(result.status, 23);
  assert.deepEqual(JSON.parse(result.stdout), [...prefix, '--version']);
});
