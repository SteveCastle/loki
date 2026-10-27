// Tests run with every build-time feature on (src/renderer/features.ts);
// a test of a disabled feature clears the variable and re-imports the module
// under jest.isolateModules. An empty value counts as unset: CI passes the
// LOKI_DREAM repository variable through, which is "" when it is not defined.
process.env.LOKI_DREAM = process.env.LOKI_DREAM || '1';
