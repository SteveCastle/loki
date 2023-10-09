import { useContext } from 'react';
import { GlobalStateContext } from '../../state';
import './generate-transcript.css';

type Props = {
  path: string;
};

export default function GenerateTranscript({ path }: Props) {
  const { libraryService } = useContext(GlobalStateContext);
  return (
    <div className="GenerateTranscript">
      <button
        className="generate"
        onClick={() => {
          libraryService.send('CREATE_JOB', {
            paths: [path],
            jobType: 'generateTranscript',
            invalidations: [['transcript']],
          });
        }}
      >
        Generate Transcript
      </button>
    </div>
  );
}
