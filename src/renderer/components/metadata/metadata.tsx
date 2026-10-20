import { useContext, useMemo, useState } from 'react';
import { store } from '../../platform';
import { useSelector } from '@xstate/react';
import { getFileType, FileTypes } from '../../../file-types';
import { GlobalStateContext } from '../../state';
import filter from '../../filter';
import eyeOpen from '../../../../assets/eye-open.svg';
import eyeClosed from '../../../../assets/eye-closed.svg';
import FileMetadata from './file-metadata';
import Duplicates from './duplicates';
import Thumbnails from './thumbnails';
import Transcript from './transcript';
import Transformations from './transformations';
import JobQueue from './job-queue';
import useJobServerAvailable from '../../hooks/useJobServerAvailable';
import './metadata.css';

type ActionMap = {
  [key: string]: JSX.Element;
};

const tabs = [
  {
    key: 'tags',
    label: 'File Info',
    fileTypes: [FileTypes.Image, FileTypes.Audio, FileTypes.Video],
    actions: [],
    active: true,
  },
  {
    key: 'transcript',
    label: 'Transcript',
    fileTypes: [FileTypes.Audio, FileTypes.Video],
    actions: ['followMetadata'],
    active: true,
  },
  {
    key: 'thumbnails',
    label: 'Thumbnails',
    actions: [],
    fileTypes: [FileTypes.Image, FileTypes.Video],
    active: true,
  },
  {
    key: 'transformations',
    label: 'Transformations',
    actions: [],
    fileTypes: [FileTypes.Audio, FileTypes.Video],
    active: false,
  },
  {
    key: 'duplicates',
    label: 'Duplicates',
    actions: [],
    fileTypes: [FileTypes.Image, FileTypes.Audio, FileTypes.Video],
    active: true,
  },
  {
    // Whole-queue management; only offered while the job server is reachable.
    key: 'jobs',
    label: 'Jobs',
    actions: [],
    fileTypes: [FileTypes.Image, FileTypes.Audio, FileTypes.Video],
    active: true,
  },
];

export default function Metadata() {
  const [activeTabKey, setActiveTabKey] = useState<string>(() => {
    // Prefer key-based persistence; fall back to old numeric index if present
    const storedKey = store.get(
      'metadata.activeTabKey',
      ''
    ) as string;
    if (storedKey) return storedKey;
    const legacyIndex = store.get(
      'metadata.activeTab',
      0
    ) as number;
    const legacyTab = tabs[legacyIndex]?.key || 'tags';
    return legacyTab;
  });
  const { libraryService } = useContext(GlobalStateContext);
  const followTranscript = useSelector(
    libraryService,
    (state) => state.context.settings.followTranscript
  );
  const toggleFollowTranscript = () => {
    libraryService.send('CHANGE_SETTING', {
      data: { followTranscript: !followTranscript },
    });
  };

  const authToken = useSelector(
    libraryService,
    (state) => state.context.authToken
  );
  const jobServerAvailable = useJobServerAvailable(authToken);

  const item = useSelector(
    libraryService,
    (state) =>
      filter(
        state.context.libraryLoadId,
        state.context.textFilter,
        state.context.library,
        state.context.settings.filters,
        state.context.settings.sortBy
      )[state.context.cursor],
    (a, b) => a?.path === b?.path
  );

  const itemType = getFileType(item?.path || '');

  // The tab the user last picked is remembered as-is; when it doesn't apply
  // (no transcript for an image, Jobs while the server check is pending or
  // before an item has loaded) the first available tab shows instead, without
  // overwriting the choice, so it comes back as soon as it applies again.
  const handleTabChange = (key: string) => {
    setActiveTabKey(key);
    store.set('metadata.activeTabKey', key);
  };

  const filteredTabs = useMemo(() => {
    return tabs.filter(
      (t) =>
        t.active &&
        t.fileTypes.includes(itemType) &&
        (t.key !== 'jobs' || jobServerAvailable === true)
    );
  }, [itemType, jobServerAvailable]);

  const shownTabKey = filteredTabs.some((t) => t.key === activeTabKey)
    ? activeTabKey
    : filteredTabs[0]?.key || 'tags';

  const getAction = (action: string) => {
    const actions: ActionMap = {
      followMetadata: (
        <button
          onClick={toggleFollowTranscript}
          className={'follow'}
          key={action}
        >
          {followTranscript ? <img src={eyeOpen} /> : <img src={eyeClosed} />}
        </button>
      ),
    };
    return actions[action];
  };

  return (
    <div className={`Metadata`}>
      <div className={`tabs`}>
        {filteredTabs.map((tab) => (
          <div
            className={`tab ${shownTabKey === tab.key ? 'active' : ''}`}
            onClick={() => handleTabChange(tab.key)}
            key={tab.key}
          >
            <span>{tab.label}</span>
            {tab.actions.map((action) => getAction(action))}
          </div>
        ))}
      </div>
      <div className={`tab-content`}>
        {shownTabKey === 'tags' && item && <FileMetadata item={item} />}
        {shownTabKey === 'thumbnails' && item && (
          <div className="thumbnails-container">
            <Thumbnails path={item.path} />
          </div>
        )}
        {shownTabKey === 'transcript' && <Transcript />}
        {shownTabKey === 'jobs' && <JobQueue />}
        {shownTabKey === 'transformations' && <Transformations />}
        {shownTabKey === 'duplicates' && item && (
          <div className="thumbnails-container">
            <Duplicates basePath={item.path} />
          </div>
        )}
      </div>
    </div>
  );
}
