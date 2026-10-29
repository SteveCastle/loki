import { getJobTitle } from '../renderer/components/controls/toast-system';

const job = (command: string, args: string[] = [], input = 'C:/a.png') =>
  ({ id: 'j', command, arguments: args, input, state: 'pending', created_at: '' } as Parameters<typeof getJobTitle>[0]);

describe('getJobTitle for the AI transform tasks', () => {
  it('names retouch jobs by what they do', () => {
    expect(getJobTitle(job('retouch', ['--preset', 'restore']))).toBe('Restoring Image');
    expect(getJobTitle(job('retouch', ['--preset', 'upscale', '--scale', '2']))).toBe('Upscaling Image');
    expect(getJobTitle(job('retouch', ['--preset', '4kify-phone']))).toBe('Making Wallpaper');
    expect(getJobTitle(job('retouch', ['--prompt', 'make it night']))).toBe('Editing Image');
    expect(getJobTitle(job('retouch', ['--prompt', 'x', '--combine', 'true']))).toBe('Combining Images');
    expect(getJobTitle(job('4kify', ['--preset', 'restore']))).toBe('Restoring Image');
  });

  it('names reshoot jobs', () => {
    expect(getJobTitle(job('reshoot', ['--animate', 'true']))).toBe('Bringing Photo to Life');
    expect(getJobTitle(job('reshoot', ['--prompt', 'x']))).toBe('Generating Video');
  });

  it('names dream jobs by mode', () => {
    expect(getJobTitle(job('dream', ['--prompt', 'a fox'], ''))).toBe('Generating Image');
    expect(getJobTitle(job('dream', ['--prompt', 'a fox', '--count', '4'], ''))).toBe('Generating Images');
    expect(getJobTitle(job('dream', ['--prompt', 'oil painting', '--strength', '0.8']))).toBe('Reimagining Image');
    expect(getJobTitle(job('dream', ['--prompt', 'sky', '--mask', 'C:/m.png']))).toBe('Inpainting Image');
  });
});
