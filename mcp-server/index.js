#!/usr/bin/env node

/**
 * MCP Server wrapper for Cerebras + Firecrawl Deep Research Agent
 *
 * This script spawns the Python MCP server and pipes STDIO.
 *
 * Usage:
 *   npx @open-deep-research/cerebras-firecrawl-mcp
 *
 * Or add to Claude Desktop config:
 *   {
 *     "mcpServers": {
 *       "cerebras-research": {
 *         "command": "npx",
 *         "args": ["@open-deep-research/cerebras-firecrawl-mcp"],
 *         "env": {
 *           "FIRECRAWL_API_KEY": "your-key",
 *           "OPENROUTER_API_KEY": "your-key"
 *         }
 *       }
 *     }
 *   }
 */

const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

// Find Python executable
function findPython() {
    const candidates = ['python3', 'python', 'py'];

    for (const cmd of candidates) {
        try {
            const result = require('child_process').spawnSync(cmd, ['--version'], {
                stdio: 'pipe',
                shell: process.platform === 'win32'
            });
            if (result.status === 0) {
                return cmd;
            }
        } catch (e) {
            continue;
        }
    }

    console.error('Error: Python not found. Please install Python 3.10+');
    process.exit(1);
}

// Find the Python package location
function findPackageDir() {
    // Check if we're running from the repo
    const repoPath = path.join(__dirname, '..', 'src', 'open_deep_research');
    if (fs.existsSync(repoPath)) {
        return path.join(__dirname, '..');
    }

    // Otherwise, assume it's installed as a package
    return null;
}

function main() {
    const python = findPython();
    const packageDir = findPackageDir();

    let args;
    let options = {
        stdio: ['pipe', 'pipe', 'inherit'],
        env: { ...process.env },
        shell: process.platform === 'win32'
    };

    if (packageDir) {
        // Running from repo - use PYTHONPATH
        options.env.PYTHONPATH = path.join(packageDir, 'src');
        options.cwd = packageDir;
        args = ['-m', 'open_deep_research.mcp_stdio_server'];
    } else {
        // Running from installed package
        args = ['-m', 'open_deep_research.mcp_stdio_server'];
    }

    // Spawn the Python process
    const pythonProcess = spawn(python, args, options);

    // Pipe stdin to Python process
    process.stdin.pipe(pythonProcess.stdin);

    // Pipe Python stdout to our stdout
    pythonProcess.stdout.pipe(process.stdout);

    // Handle process events
    pythonProcess.on('error', (err) => {
        console.error(`Failed to start Python server: ${err.message}`);
        process.exit(1);
    });

    pythonProcess.on('exit', (code, signal) => {
        if (signal) {
            console.error(`Python server killed by signal: ${signal}`);
        }
        process.exit(code || 0);
    });

    // Handle our own termination
    process.on('SIGINT', () => {
        pythonProcess.kill('SIGINT');
    });

    process.on('SIGTERM', () => {
        pythonProcess.kill('SIGTERM');
    });
}

main();
