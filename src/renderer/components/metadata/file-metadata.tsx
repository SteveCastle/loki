import { useQuery, useQueryClient } from '@tanstack/react-query';
import { Metadata } from '../../../main/metadata';

import './file-metadata.css';
import InfoTable from './info-table';
import Tags from './tags';
import PathTree from './path-tree';
import Skeleton, { SkeletonTheme } from 'react-loading-skeleton';

const loadFileMetadata = (path: string) => async (): Promise<Metadata> => {
  let metadata: any;
  metadata = await window.electron.ipcRenderer.invoke('load-file-metadata', [
    path,
  ]);

  metadata = metadata || { path, tags: '' };
  return metadata as Metadata;
};

export default function FileMetadata({ item }: { item: any }) {
  const queryClient = useQueryClient();
  const path = item?.path;
  const { data, error, isLoading } = useQuery<Metadata, Error>(
    ['file-metadata', path],
    loadFileMetadata(path),
    { retry: true }
  );
  if (isLoading || !data)
    return (
      <div className="FileMetadata">
        <div className="section">
          <h2>Path</h2>
          {item?.path && <PathTree path={item?.path} />}
        </div>
        <div className="section">
          <h2>Tags</h2>
          {item?.path && <Tags item={item} />}
        </div>
        <div className="placeholder">
          <SkeletonTheme baseColor="#202020" highlightColor="#444">
            <Skeleton count={3} />
          </SkeletonTheme>
        </div>
        <div className="placeholder">
          <SkeletonTheme baseColor="#202020" highlightColor="#444">
            <Skeleton count={3} />
          </SkeletonTheme>
        </div>
      </div>
    );
  if (error) return <p>{error.message}</p>;
  return (
    <div className="FileMetadata">
      <div className="section">
        <h2>Path</h2>
        {item?.path && <PathTree path={item?.path} />}
      </div>
      <div className="section">
        <h2>Tags</h2>
        {item?.path && <Tags item={item} />}
      </div>
      <div className="section">
        <h2
          onClick={() => {
            const copyContent = async (text: string) => {
              try {
                await navigator.clipboard.writeText(text);
                console.log('Content copied to clipboard');
              } catch (err) {
                console.error('Failed to copy: ', err);
              }
            };
            const allTags: string[] = [];
            if (data.extendedMetadata?.tags) {
              const tags = Object.values(data.extendedMetadata?.tags);
              allTags.push(...tags.flat());
            }
            console.log(allTags);
            const htmlString = allTags.join(', ');
            // convert html encoded string to plain text
            const plainText = decodeURIComponent(htmlString);

            copyContent(plainText);
          }}
        >
          Suggested Tags
        </h2>
        <div className={``}>
          {data.extendedMetadata?.tags &&
            Object.keys(data.extendedMetadata?.tags).map((category: string) => (
              <div className="tag-category" key={category}>
                <h3>{category}</h3>
                <div className="tag-list">
                  {data.extendedMetadata?.tags[category].map((tag: string) => (
                    <span
                      key={tag}
                      className="tag"
                      onClick={() => {
                        async function createAssignment() {
                          await window.electron.ipcRenderer.invoke(
                            'create-assignment',
                            [[item.path], tag, category, 0, false]
                          );
                          queryClient.invalidateQueries({
                            queryKey: ['metadata'],
                          });
                          queryClient.invalidateQueries({
                            queryKey: ['taxonomy'],
                          });
                        }
                        createAssignment();
                      }}
                    >
                      {tag}
                    </span>
                  ))}
                </div>
              </div>
            ))}
        </div>
      </div>
      <div className="section">
        <h2>File Metadata</h2>
        <div className={``}>
          <InfoTable data={data.fileMetadata} />
        </div>
      </div>

      {data.stableDiffusionMetaData && (
        <div className="section">
          <h2>Stable Diffusion</h2>
          <div className={``}>
            <InfoTable data={data.stableDiffusionMetaData} />
          </div>
        </div>
      )}
    </div>
  );
}
