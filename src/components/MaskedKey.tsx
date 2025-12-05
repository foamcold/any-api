import { useState, useLayoutEffect, useRef } from 'react';
import { Eye, EyeOff } from 'lucide-react';

interface MaskedKeyProps {
  apiKey: string;
}

export default function MaskedKey({ apiKey }: MaskedKeyProps) {
  const [isHovered, setIsHovered] = useState(false);
  const [width, setWidth] = useState(0);
  const formattedKeyRef = useRef<HTMLElement>(null);

  const formatKey = (key: string) => {
    if (!key || key.length <= 8) {
      return key;
    }
    return `${key.substring(0, 4)}...${key.substring(key.length - 4)}`;
  };

  useLayoutEffect(() => {
    if (!apiKey) return;
    // We render the actual key once off-screen to measure its width.
    const tempElement = document.createElement('code');
    tempElement.className = "relative rounded bg-muted px-[0.3rem] py-[0.2rem] font-mono text-sm break-all";
    tempElement.style.position = 'absolute';
    tempElement.style.visibility = 'hidden';
    tempElement.style.whiteSpace = 'nowrap'; // Ensure it doesn't wrap
    tempElement.innerText = apiKey;
    document.body.appendChild(tempElement);
    setWidth(tempElement.offsetWidth);
    document.body.removeChild(tempElement);
  }, [apiKey]);

  if (!apiKey) {
    return <span>-</span>;
  }

  return (
    <div
      className="flex items-center gap-2"
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      <code
        ref={formattedKeyRef}
        className="relative rounded bg-muted px-[0.3rem] py-[0.2rem] font-mono text-sm break-all"
        title={apiKey}
        style={{ minWidth: `${width}px`, display: 'inline-block' }}
      >
        {isHovered ? apiKey : formatKey(apiKey)}
      </code>
      {isHovered ? <EyeOff className="w-3 h-3" /> : <Eye className="w-3 h-3" />}
    </div>
  );
}