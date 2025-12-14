/**
 * Shrike Chrome Extension Icon Generator
 *
 * This script generates PNG icons for the Chrome extension.
 * Run with: node generate-icons.js
 *
 * Note: This creates simple placeholder icons. For production,
 * consider using the project's logo.ico or custom designed icons.
 */

const fs = require("fs");
const path = require("path");

// Simple PNG generator for placeholder icons
// Creates a minimal valid PNG with a colored square

function createPNG(size, color = { r: 255, g: 107, b: 53 }) {
  // PNG signature
  const signature = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);

  // IHDR chunk (image header)
  const ihdrData = Buffer.alloc(13);
  ihdrData.writeUInt32BE(size, 0); // Width
  ihdrData.writeUInt32BE(size, 4); // Height
  ihdrData.writeUInt8(8, 8); // Bit depth
  ihdrData.writeUInt8(2, 9); // Color type (RGB)
  ihdrData.writeUInt8(0, 10); // Compression
  ihdrData.writeUInt8(0, 11); // Filter
  ihdrData.writeUInt8(0, 12); // Interlace

  const ihdrChunk = createChunk("IHDR", ihdrData);

  // IDAT chunk (image data) - uncompressed for simplicity
  // We'll create a simple gradient/shape
  const rawData = [];
  const centerX = size / 2;
  const centerY = size / 2;
  const maxDist = size * 0.4;

  for (let y = 0; y < size; y++) {
    rawData.push(0); // Filter byte (none)
    for (let x = 0; x < size; x++) {
      // Create a simple hexagonal/diamond shape
      const dx = Math.abs(x - centerX);
      const dy = Math.abs(y - centerY);
      const dist = dx + dy;

      if (dist < maxDist) {
        // Inside shape - use accent color with gradient
        const intensity = 1 - (dist / maxDist) * 0.3;
        rawData.push(Math.floor(color.r * intensity));
        rawData.push(Math.floor(color.g * intensity));
        rawData.push(Math.floor(color.b * intensity));
      } else if (dist < maxDist + 2) {
        // Border
        rawData.push(Math.floor(color.r * 0.7));
        rawData.push(Math.floor(color.g * 0.7));
        rawData.push(Math.floor(color.b * 0.7));
      } else {
        // Background - dark
        rawData.push(22);
        rawData.push(22);
        rawData.push(22);
      }
    }
  }

  // Compress with zlib (deflate)
  const zlib = require("zlib");
  const compressed = zlib.deflateSync(Buffer.from(rawData), { level: 9 });
  const idatChunk = createChunk("IDAT", compressed);

  // IEND chunk (end)
  const iendChunk = createChunk("IEND", Buffer.alloc(0));

  return Buffer.concat([signature, ihdrChunk, idatChunk, iendChunk]);
}

function createChunk(type, data) {
  const length = Buffer.alloc(4);
  length.writeUInt32BE(data.length, 0);

  const typeBuffer = Buffer.from(type, "ascii");
  const crcData = Buffer.concat([typeBuffer, data]);
  const crc = crc32(crcData);

  const crcBuffer = Buffer.alloc(4);
  crcBuffer.writeUInt32BE(crc >>> 0, 0);

  return Buffer.concat([length, typeBuffer, data, crcBuffer]);
}

// CRC32 implementation for PNG
function crc32(data) {
  let crc = 0xffffffff;
  const table = getCRC32Table();

  for (let i = 0; i < data.length; i++) {
    crc = table[(crc ^ data[i]) & 0xff] ^ (crc >>> 8);
  }

  return crc ^ 0xffffffff;
}

let crcTable = null;
function getCRC32Table() {
  if (crcTable) return crcTable;

  crcTable = new Uint32Array(256);
  for (let i = 0; i < 256; i++) {
    let c = i;
    for (let j = 0; j < 8; j++) {
      c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    }
    crcTable[i] = c;
  }
  return crcTable;
}

// Generate icons
const sizes = [16, 32, 48, 128];
const iconsDir = path.join(__dirname, "icons");

// Ensure icons directory exists
if (!fs.existsSync(iconsDir)) {
  fs.mkdirSync(iconsDir, { recursive: true });
}

console.log("Generating Shrike extension icons...");

sizes.forEach((size) => {
  const filename = `icon${size}.png`;
  const filepath = path.join(iconsDir, filename);

  try {
    const pngData = createPNG(size);
    fs.writeFileSync(filepath, pngData);
    console.log(`✓ Created ${filename} (${size}x${size})`);
  } catch (err) {
    console.error(`✗ Failed to create ${filename}:`, err.message);
  }
});

console.log("\nDone! Icons created in the icons/ folder.");
console.log("You can now load the extension in Chrome.");
