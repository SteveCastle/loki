import React from 'react';
import { FileMetadata, StableDiffusionMetaData } from '../../../main/metadata';
import copyIcon from '../../../../assets/copy.svg';

import './info-table.css';

// This used to be moment(...).format('MMMM Do YYYY, h:mm:ss a'). Moment is
// 688KB of source — the second-largest dependency in the renderer — and the
// whole app was parsing it at startup to render one timestamp in a panel that
// starts collapsed. Intl does the same job with what the platform already has.

const dateStyle = new Intl.DateTimeFormat(undefined, {
  month: 'long',
  day: 'numeric',
  year: 'numeric',
});
const timeStyle = new Intl.DateTimeFormat(undefined, {
  hour: 'numeric',
  minute: '2-digit',
  second: '2-digit',
  hour12: true,
});

// Keeps moment's "August 8th 2026" ordinal so the panel reads the same as before.
function ordinal(day: number): string {
  const rem100 = day % 100;
  if (rem100 >= 11 && rem100 <= 13) return `${day}th`;
  switch (day % 10) {
    case 1:
      return `${day}st`;
    case 2:
      return `${day}nd`;
    case 3:
      return `${day}rd`;
    default:
      return `${day}th`;
  }
}

// `modified` arrives as a Date over Electron IPC and as an ISO string over
// HTTP, so accept either. Anything unparseable falls through to the raw value.
function formatDate(value: unknown): string | null {
  const date =
    value instanceof Date
      ? value
      : typeof value === 'string' && !Number.isNaN(Date.parse(value))
      ? new Date(value)
      : null;
  if (!date || Number.isNaN(date.getTime())) return null;
  const parts = dateStyle.formatToParts(date);
  const month = parts.find((p) => p.type === 'month')?.value ?? '';
  const year = parts.find((p) => p.type === 'year')?.value ?? '';
  return `${month} ${ordinal(date.getDate())} ${year}, ${timeStyle
    .format(date)
    .toLowerCase()}`;
}

type Props = {
  data: FileMetadata | StableDiffusionMetaData;
};

const InfoTable: React.FC<Props> = ({ data }) => {
  if (!data) return null;
  return (
    <div className="InfoTable">
      <table>
        <tbody>
          {Object.entries(data).map(([key, value]) => {
            let formattedValue;
            if (typeof value === 'boolean') {
              formattedValue = value ? '✔️' : '❌';
            } else if (key === 'modified' && formatDate(value)) {
              formattedValue = formatDate(value);
            } else {
              formattedValue = value;
            }

            return (
              <tr key={key}>
                <td style={{ fontWeight: 'bold', textTransform: 'capitalize' }}>
                  {key}
                </td>
                <td>
                  <div className="value">
                    {formattedValue}
                    <button
                      className="copy-button"
                      onClick={() => {
                        const copyContent = async (text: string) => {
                          try {
                            await navigator.clipboard.writeText(text);
                            console.log('Content copied to clipboard');
                          } catch (err) {
                            console.error('Failed to copy: ', err);
                          }
                        };
                        copyContent(value);
                      }}
                    >
                      <img src={copyIcon} />
                    </button>
                  </div>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
};

export default InfoTable;
