---
name: remote-container-workspace
description: Connect to remote SSH hosts, verify Docker or compatible containers, run commands inside containers, and open an interactive shell in a requested container working directory. Use when a user asks Codex to connect to a remote machine, enter a named container, adopt a container path as the working directory, execute or inspect remote development jobs, or repeat this workflow across hosts. Prefer SSH keys and never persist passwords in the skill, scripts, shell history, or configuration.
---

# Remote Container Workspace

Connect to an SSH host, validate the requested container workspace, and perform remote work with explicit safety boundaries.

## Gather connection parameters

Determine these values from the request or existing task context:

- SSH host, user, and optional port
- Authentication method: SSH agent/key preferred; password only through a temporary environment variable or an interactive prompt
- Container name
- Absolute working directory inside the container
- Whether the user wants a one-shot command or an interactive terminal

Ask only for missing values that cannot be inferred safely. Never copy a password into `SKILL.md`, scripts, profiles, command arguments, logs, or saved configuration. If the user supplied a password in chat, use it only for the current connection and recommend migrating to an SSH key.

## Connect and verify

1. Obtain network permission when the execution environment requires it.
2. Check that an SSH client is available.
3. Connect using host-key verification. Use `StrictHostKeyChecking=accept-new` for a first connection unless the user requires a pre-provisioned fingerprint. Never use `StrictHostKeyChecking=no`.
4. Run a non-interactive validation before doing work:

   ```text
   docker exec -w <workdir> <container> pwd
   ```

5. Confirm that the returned path exactly matches the requested workspace. Diagnose missing containers or directories before continuing.
6. Run requested commands non-interactively when possible. Open a visible interactive terminal only when the user needs to control a shell.

Use `powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/connect-container.ps1 ...` on Windows for a consistent implementation. The process-level bypass supports systems that disable local scripts without changing the system execution policy. Use `-DryRun` first when parameters or quoting are uncertain.

## Windows and PowerShell command reliability

PowerShell parses pipes, parentheses, quotes, `$()`, and here-strings before SSH sees them. A command that is valid in Linux shell can fail locally with errors such as "A positional parameter cannot be found", "module could not be loaded", or unexpected token parsing. Treat these as local PowerShell quoting failures unless the remote output clearly proves the command reached the container.

Use these patterns in order:

1. Prefer the bundled script for short, simple one-shot commands:

   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
   & '<skill-dir>\scripts\connect-container.ps1' `
     -HostName <host> `
     -UserName <user> `
     -Container <container> `
     -WorkDir <workdir> `
     -Command 'pwd && git status --short'
   ```

2. For commands containing shell pipes, `grep -E`, `find -printf`, nested quotes, Python here-docs, or multi-line logic, avoid passing the logic through PowerShell as a quoted `-Command`. Pipe the script over stdin to `docker exec -i` instead:

   ```powershell
   @'
   set -eu
   pwd
   grep -nE 'pattern1|pattern2' /etc/hosts || true
   python3 - <<'PY'
   print("remote python ran inside the container")
   PY
   '@ | ssh.exe -o StrictHostKeyChecking=accept-new <user>@<host> `
     "docker exec -i -w <workdir> <container> sh"
   ```

3. For Python-heavy inspection, prefer stdin Python directly. This avoids shell quoting almost entirely:

   ```powershell
   @'
   from pathlib import Path
   p = Path("/remote/path")
   print(p.exists(), p)
   '@ | ssh.exe -o StrictHostKeyChecking=accept-new <user>@<host> `
     "docker exec -i -w <workdir> <container> python3 -"
   ```

4. Do not combine complex remote shell syntax with an outer `powershell.exe -File ... -Command <string>` invocation. If a script call is needed, invoke the `.ps1` directly from the current PowerShell process with `&`, or switch to stdin SSH for complex commands.

When diagnosing a failure, first decide where it failed:

- Local PowerShell failure: mentions positional parameters, modules, unexpected tokens before SSH output, or shows pieces of the intended Linux command as PowerShell errors.
- SSH or host failure: SSH authentication, host key, network, or `docker` errors.
- Container command failure: output includes the container working directory or command stderr from inside Linux.

Keep SSH host-key verification enabled with `StrictHostKeyChecking=accept-new`; do not use `StrictHostKeyChecking=no` to work around quoting problems.

## Execute remote work

- For a one-shot command, execute it with `docker exec -w <workdir> <container> sh -lc <command>`.
- For an interactive session, allocate a TTY and execute `docker exec -it -w <workdir> <container> <shell>`.
- Treat remote commands as changes to a live external system. Stay within the user-requested host, container, path, and task.
- Do not start destructive operations, modify unrelated services, or broaden the target without explicit authorization.
- Report the verified host, container, working directory, and command outcome. Do not echo credentials.

## Authentication

Prefer, in order:

1. SSH agent or default key
2. Explicit identity file
3. Interactive password entry
4. A password stored only in a process environment variable for the current operation

The bundled script accepts `-PasswordEnvironmentVariable`. It copies that value into a short-lived child-process environment variable and creates only a temporary credential-free askpass helper. It removes the helper after SSH exits.

## Example

Verify the known development workspace without storing its password:

```powershell
.\scripts\connect-container.ps1 `
  -HostName 10.20.35.29 `
  -UserName mccxadmin `
  -Container jd_dev `
  -WorkDir /data/bevformer_bk/liang.geng/jd_test
```

Add `-Interactive` to open a shell, or add `-Command 'git status --short'` for a one-shot command.
