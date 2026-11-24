// news-image-20261123 entry program (Node 22, standard library only).
// Reads the rendered Monday 23 November 2026 world news front page (entry.png)
// from the repository root and writes it deterministically to /output/entry.png.

import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const SOURCE_PATH = fileURLToPath(new URL('./entry.png', import.meta.url));
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
const TARGET_PATH = join(OUTPUT_DIR, 'entry.png');

const PNG_SIGNATURE = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
const MAX_BYTES = 1024 * 1024;
const MAX_DIMENSION = 1024;

function assertValidPng(bytes: Buffer): void {
  if (bytes.byteLength < 24 || bytes.byteLength > MAX_BYTES) {
    throw new Error(`Invalid entry.png size: ${bytes.byteLength} bytes (max ${MAX_BYTES})`);
  }
  if (!bytes.subarray(0, PNG_SIGNATURE.length).equals(PNG_SIGNATURE)) {
    throw new Error('entry.png does not start with a valid PNG signature');
  }
  const chunkType = bytes.subarray(12, 16).toString('ascii');
  if (chunkType !== 'IHDR') {
    throw new Error(`Expected IHDR chunk first, found ${chunkType}`);
  }
  const width = bytes.readUInt32BE(16);
  const height = bytes.readUInt32BE(20);
  if (width === 0 || height === 0 || width > MAX_DIMENSION || height > MAX_DIMENSION) {
    throw new Error(`Invalid entry.png dimensions: ${width}x${height} (max ${MAX_DIMENSION}x${MAX_DIMENSION})`);
  }
}

function exportEntryImage(): void {
  const bytes = readFileSync(SOURCE_PATH);
  assertValidPng(bytes);
  mkdirSync(dirname(TARGET_PATH), { recursive: true });
  writeFileSync(TARGET_PATH, bytes);
}

exportEntryImage();
