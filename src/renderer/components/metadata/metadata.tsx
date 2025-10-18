import { useContext, useEffect, useMemo, useState } from 'react';
import { useSelector } from '@xstate/react';
import { getFileType, FileTypes } from '../../../file-types';
import { GlobalStateContext } from '../../state';
import filter from '../../filter';
import eyeOpen from '../../../../assets/eye-open.svg';
import eyeClosed from '../../../../assets/eye-closed.svg';
import FileMetadata from './file-metadata';
import Duplicates from './duplicates';
import Transcript from './transcript';
import Transformations from './transformations';
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
];

export default function Metadata() {
  const [activeTabKey, setActiveTabKey] = useState<string>(() => {
    // Prefer key-based persistence; fall back to old numeric index if present
    const storedKey = window.electron.store.get(
      'metadata.activeTabKey',
      ''
    ) as string;
    if (storedKey) return storedKey;
    const legacyIndex = window.electron.store.get(
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

  const handleTabChange = (key: string) => {
    setActiveTabKey(key);
    window.electron.store.set('metadata.activeTabKey', key);
  };

  const filteredTabs = useMemo(() => {
    return tabs.filter((t) => t.active && t.fileTypes.includes(itemType));
  }, [itemType]);

  // Ensure active tab remains valid across type changes; default to first available
  useEffect(() => {
    if (!filteredTabs.find((t) => t.key === activeTabKey)) {
      const nextKey = filteredTabs[0]?.key || 'tags';
      handleTabChange(nextKey);
    }
  }, [filteredTabs, activeTabKey, handleTabChange]);

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
            className={`tab ${activeTabKey === tab.key ? 'active' : ''}`}
            onClick={() => handleTabChange(tab.key)}
            key={tab.key}
          >
            <span>{tab.label}</span>
            {tab.actions.map((action) => getAction(action))}
          </div>
        ))}
      </div>
      <div className={`tab-content`}>
        {activeTabKey === 'tags' && item && <FileMetadata item={item} />}
        {activeTabKey === 'transcript' && <Transcript />}
        {activeTabKey === 'transformations' && <Transformations />}
        {activeTabKey === 'duplicates' && item && (
          <div style={{ height: '100%', overflow: 'hidden' }}>
            <Duplicates basePath={item.path} />
          </div>
        )}
      </div>
    </div>
  );
}
