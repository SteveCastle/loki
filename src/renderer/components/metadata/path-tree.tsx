import { FC } from 'react';
import { useContext } from 'react';
import { GlobalStateContext } from '../../state';
import path from 'path-browserify';
import './path-tree.css';

interface PathTreeProps {
  path: string;
}

const PathTree: FC<PathTreeProps> = ({ path: pathname }) => {
  // Normalize the path to correct any Windows/Unix discrepancies
  const { libraryService } = useContext(GlobalStateContext);
  const normalizedPath = path.normalize(pathname);
  const pathParts = normalizedPath.split('\\');

  // Recursive function to generate the tree structure
  const renderPathTree = (parts: string[], index = 0): JSX.Element | null => {
    if (index >= parts.length || !parts[index]) {
      return null;
    }

    // Set the path up until this point in a variable called fullPath.
    // This is used to set the path when the user clicks on a folder.

    if (index === parts.length - 1) {
      return (
        <ul>
          <li>
            {parts[index]}
            {renderPathTree(parts, index + 1)}
          </li>
        </ul>
      );
    }
    const fullPath = parts.slice(0, index + 2).join('\\');

    return (
      <ul>
        <li
          onClick={() => libraryService.send('SET_FILE', { path: fullPath })}
          className="pathLink"
        >
          {parts[index]}
          {renderPathTree(parts, index + 1)}
        </li>
      </ul>
    );
  };

  return <div className="PathTree">{renderPathTree(pathParts)}</div>;
};

export default PathTree;
