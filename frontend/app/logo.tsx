export function Logo({ size = 20 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor" aria-hidden>
      <rect x="4.5" y="4.5" width="15" height="2.4" rx="1.2" />
      <rect x="4.5" y="17.1" width="15" height="2.4" rx="1.2" />
      <rect x="7.4" y="8.4" width="2.2" height="7.2" rx="1.1" />
      <rect x="10.9" y="8.4" width="2.2" height="7.2" rx="1.1" />
      <rect x="14.4" y="8.4" width="2.2" height="7.2" rx="1.1" />
    </svg>
  );
}
