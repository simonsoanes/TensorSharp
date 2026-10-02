#!/usr/bin/env node
// Native Node entry point: Windows must not send a desktop browser into WSL.
import { spawn } from 'node:child_process';
import { randomUUID } from 'node:crypto';
import { closeSync, existsSync, mkdirSync, openSync, realpathSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const packageArgs = ['--yes', '--prefer-offline', '--package', '@playwright/cli@0.1.21', 'playwright-cli'];

export function parseOptions(argv) {
  const result = { session: false, filename: false, help: false, version: false, command: '' };
  let skipValue = false;
  let booleanOption = '';
  let literalArgs = false;
  for (const arg of argv) {
    if (booleanOption) {
      if (arg === 'true' || arg === 'false') {
        if (booleanOption !== 'other') result[booleanOption] = arg === 'true';
        booleanOption = '';
        continue;
      }
      booleanOption = '';
    }
    if (skipValue) { skipValue = false; continue; }
    if (literalArgs) {
      if (!result.command) result.command = arg;
      continue;
    }
    const equals = arg.indexOf('=');
    const flag = equals < 0 ? arg : arg.slice(0, equals);
    if (flag === '--session' || flag === '-s' || flag === '--filename') {
      result[flag === '--filename' ? 'filename' : 'session'] = true;
      skipValue = equals < 0;
    } else if (['--depth', '--config', '--browser', '--device', '--idle-timeout', '--profile'].includes(flag)) {
      skipValue = equals < 0;
    } else if (['--json', '--raw', '--boxes', '--headed', '--persistent', '--mobile'].includes(flag)) {
      if (equals < 0) booleanOption = 'other';
    } else if (['--help', '-h', '--version', '-v'].includes(flag)) {
      const option = flag === '--help' || flag === '-h' ? 'help' : 'version';
      result[option] = equals < 0 || arg.slice(equals + 1) !== 'false';
      if (equals < 0) booleanOption = option;
    } else if (arg === '--') {
      literalArgs = true;
    } else if (!arg.startsWith('-') && !result.command) {
      result.command = arg;
    }
  }
  return result;
}

export function buildCliArgs(argv, { env = process.env, cwd = process.cwd() } = {}) {
  const options = parseOptions(argv);
  const args = [...packageArgs];
  if (!options.session && env.PLAYWRIGHT_CLI_SESSION) {
    args.push('--session', env.PLAYWRIGHT_CLI_SESSION);
  }
  // Bare snapshot otherwise prints the whole page. Reserve a unique file and
  // put its option before the caller's possible end-of-options marker.
  if (options.command === 'snapshot' && !options.filename && !options.help && !options.version) {
    const relative = `output/playwright/snapshot-${randomUUID()}.yml`;
    mkdirSync(path.join(cwd, 'output', 'playwright'), { recursive: true });
    closeSync(openSync(path.join(cwd, relative), 'wx', 0o600));
    args.push(`--filename=${relative}`);
  }
  return [...args, ...argv];
}

export function resolveNpxCli({ execPath = process.execPath, env = process.env } = {}) {
  // Official Node distributions, nvm-windows and npm's global Windows prefix
  // place this entry point alongside node.exe or an npm/npx launcher on PATH.
  // Never execute npx.cmd via cmd.exe: that would reinterpret browser arguments.
  const pathKey = Object.keys(env).find(key => key.toLowerCase() === 'path');
  const directories = [path.dirname(execPath), ...(env[pathKey] || '').split(';')];
  for (let directory of directories) {
    directory = directory.trim().replace(/^"(.*)"$/, '$1');
    if (!directory) continue;
    const candidate = path.join(directory, 'node_modules', 'npm', 'bin', 'npx-cli.js');
    if (existsSync(candidate)) return candidate;
  }
  throw new Error('Native Windows npm was not found. Install Node.js for Windows with npm, '
    + 'then restart the host so node.exe and npm are on PATH. WSL Node/npm cannot launch this desktop browser.');
}

export function createInvocation(argv, options = {}) {
  const platform = options.platform ?? process.platform;
  const execPath = options.execPath ?? process.execPath;
  // Resolve dependencies before creating snapshot artifacts on failure.
  const prefix = platform === 'win32' ? [resolveNpxCli(options)] : [];
  return {
    command: platform === 'win32' ? execPath : 'npx',
    args: [...prefix, ...buildCliArgs(argv, options)],
  };
}

export async function runCli(argv = process.argv.slice(2)) {
  const invocation = createInvocation(argv);
  const env = { ...process.env };
  if (process.platform === 'win32') {
    // npm's cached command shim must also find this native Node installation,
    // including when the host launched a portable node.exe by absolute path.
    const pathKey = Object.keys(env).find(key => key.toLowerCase() === 'path') || 'PATH';
    env[pathKey] = `${path.dirname(process.execPath)};${env[pathKey] || ''}`;
  }
  const child = spawn(invocation.command, invocation.args, {
    stdio: 'inherit', shell: false, windowsHide: true, env,
  });
  const forwardSignal = signal => { if (!child.killed) child.kill(signal); };
  const interrupt = () => forwardSignal('SIGINT');
  const terminate = () => forwardSignal('SIGTERM');
  process.on('SIGINT', interrupt);
  process.on('SIGTERM', terminate);
  try {
    return await new Promise((resolve, reject) => {
      child.once('error', reject);
      child.once('exit', (code, signal) => resolve(code ?? (signal === 'SIGINT' ? 130 : 143)));
    });
  } finally {
    process.off('SIGINT', interrupt);
    process.off('SIGTERM', terminate);
  }
}

// ESM resolves symlinks, while argv[1] can retain an installed skill's link path.
if (process.argv[1] && realpathSync(fileURLToPath(import.meta.url)) === realpathSync(process.argv[1])) {
  try {
    process.exitCode = await runCli();
  } catch (error) {
    console.error(`Error: ${error.message}`);
    process.exitCode = 1;
  }
}
