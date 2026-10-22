import './loader.css';

type LoaderProps = {
  // Shown while a folder scan streams in: stops it and keeps what loaded.
  onStop?: () => void;
};

export function Loader({ onStop }: LoaderProps) {
  return (
    <div className="Loader">
      {onStop && (
        <button
          type="button"
          className="Loader-stop"
          onClick={onStop}
          title="Stop loading this folder and keep the files found so far"
        >
          Stop loading
        </button>
      )}
      <div className="spinner"></div>
    </div>
  );
}
