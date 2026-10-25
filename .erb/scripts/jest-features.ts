// Tests run with every build-time feature on (src/renderer/features.ts);
// a test of a disabled feature clears the variable and re-imports the module
// under jest.isolateModules.
process.env.LOKI_DREAM = process.env.LOKI_DREAM ?? '1';
