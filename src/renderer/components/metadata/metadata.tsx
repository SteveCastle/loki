import { useContext, useEffect, useState } from 'react';
import { useSelector } from '@xstate/react';
import { getFileType, FileTypes } from '../../../file-types';
import { GlobalStateContext } from '../../state';
import filter from '../../filter';
import eyeOpen from '../../../../assets/eye-open.svg';
import eyeClosed from '../../../../assets/eye-closed.svg';
import FileMetadata from './file-metadata';
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
];

export default function Metadata() {
  const [activeTab, setActiveTab] = useState(() => {
    return window.electron.store.get('metadata.activeTab', 0);
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

  const handleTabChange = (tabIndex: number) => {
    setActiveTab(tabIndex);
    window.electron.store.set('metadata.activeTab', tabIndex);
  };

  useEffect(() => {
    if (itemType === FileTypes.Image) {
      handleTabChange(0);
    }
  }, [itemType]);

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
        {tabs
          .filter((t) => t.active && t.fileTypes.includes(itemType))
          .map((tab, idx) => (
            <div
              className={`tab ${activeTab === idx ? 'active' : ''}`}
              onClick={() => handleTabChange(idx)}
              key={tab.label}
            >
              <span>{tab.label}</span>
              {tab.actions.map((action) => getAction(action))}
            </div>
          ))}
      </div>
      <div className={`tab-content`}>
        {activeTab === 0 && item && <FileMetadata item={item} />}
        {activeTab === 1 && <Transcript />}
        {activeTab === 2 && <Transformations />}
      </div>
    </div>
  );
}
