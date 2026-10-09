import { getJobTitle } from '../renderer/components/controls/toast-system';

const job = (command: string, args: string[] = []) =>
  ({ id: 'j', command, arguments: args, input: 'C:/a.png', state: 'pending', created_at: '' } as Parameters<typeof getJobTitle>[0]);

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
});
