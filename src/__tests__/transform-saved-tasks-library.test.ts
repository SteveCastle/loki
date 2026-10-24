/**
 * Saved Transform tasks belong to a library: each database sees only its own,
 * and the old single list moves into one library on first load.
 */
const mockSettings: Record<string, unknown> = {};

jest.mock('../renderer/platform', () => ({
  ...jest.requireActual('../renderer/platform'),
  isElectron: true,
  store: {
    get: (k: string, d?: unknown) => (k in mockSettings ? mockSettings[k] : d),
    set: (k: string, v: unknown) => {
      mockSettings[k] = v;
    },
    getMany: () => ({}),
  },
}));

import { settingsFor } from '../renderer/components/transform/intents';
import {
  loadTemplates,
  resetTemplateCache,
  saveTemplate,
  setTemplateLibrary,
  templateLibraryKey,
} from '../renderer/components/transform/store';

const DREAM = 'C:\\Users\\me\\AppData\\Roaming\\Lowkey Media Viewer\\dream.sqlite';
const DREAM_X = 'C:\\Users\\me\\AppData\\Roaming\\Lowkey Media Viewer\\dream-x.sqlite';

const names = () => loadTemplates('edit').map((t) => t.name);

beforeEach(() => {
  window.localStorage.clear();
  for (const k of Object.keys(mockSettings)) delete mockSettings[k];
  resetTemplateCache();
});

test('one key per database file, however the path is spelled', () => {
  expect(templateLibraryKey(DREAM_X)).toBe(templateLibraryKey(DREAM_X.replace(/\\/g, '/').toUpperCase()));
  expect(templateLibraryKey(DREAM)).not.toBe(templateLibraryKey(DREAM_X));
});

test('each library sees only the tasks saved in it', () => {
  mockSettings.dbPath = DREAM_X;
  saveTemplate('edit', 'Night', 'make it night', settingsFor('edit'));
  expect(names()).toEqual(['Night']);

  setTemplateLibrary(DREAM);
  expect(names()).toEqual([]);
  saveTemplate('edit', 'Snow', 'add snow', settingsFor('edit'));
  expect(names()).toEqual(['Snow']);

  setTemplateLibrary(DREAM_X);
  expect(names()).toEqual(['Night']);

  // ...and it survives a restart.
  resetTemplateCache();
  mockSettings.dbPath = DREAM;
  expect(names()).toEqual(['Snow']);
});

const legacy = { edit: [{ id: 'a1', name: 'Old', prompt: 'old prompt' }] };

test('the old single list moves into the library open at the time', () => {
  window.localStorage.setItem('loki.transform.templates.v1', JSON.stringify(legacy));
  mockSettings.dbPath = DREAM;
  expect(names()).toEqual(['Old']);
  setTemplateLibrary(DREAM_X);
  expect(names()).toEqual([]);
  // Kept as a backup.
  expect(window.localStorage.getItem('loki.transform.templates.v1')).not.toBeNull();
});

test('...or into the library named by transformTemplatesLegacyLibrary', () => {
  window.localStorage.setItem('loki.transform.templates.v1', JSON.stringify(legacy));
  mockSettings.dbPath = DREAM;
  mockSettings.transformTemplatesLegacyLibrary = DREAM_X;
  expect(names()).toEqual([]);
  setTemplateLibrary(DREAM_X);
  expect(names()).toEqual(['Old']);

  // Moved once: a later change to the setting moves nothing.
  resetTemplateCache();
  mockSettings.transformTemplatesLegacyLibrary = DREAM;
  mockSettings.dbPath = DREAM;
  expect(names()).toEqual([]);
});
