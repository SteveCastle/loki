import { useQuery } from '@tanstack/react-query';
import './taxonomy.css';
type Concept = {
  label: string;
  weight: number;
  category: string;
};

type Props = {
  tag: Concept;
};

const fetchTagCount = (tag: string) => async (): Promise<number> => {
  const count = await window.electron.fetchTagCount(tag);
  return count;
};

export default function TagCount({ tag }: Props) {
  const { data: count } = useQuery<number, Error>(
    ['taxonomy', 'tag', tag.label, 'count'],
    fetchTagCount(tag.label),
    {
      refetchOnWindowFocus: false,
    }
  );

  return <span>{count}</span>;
}
