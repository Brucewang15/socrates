// A Doric column: entablature, fluted shaft, flared base. Drawn on a 24-unit
// grid so it stays crisp at the 16-18px it is actually used at.
export function Logo({ size = 20 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor" aria-hidden>
      {/* abacus, then the echinus tapering into the shaft */}
      <rect x="2.6" y="2.8" width="18.8" height="2.5" rx="1" />
      <path d="M5.6 5.8h12.8l-1.5 1.9H7.1z" />
      {/* three flutes */}
      <rect x="7.5" y="8.3" width="1.9" height="9.6" rx="0.95" />
      <rect x="11.05" y="8.3" width="1.9" height="9.6" rx="0.95" />
      <rect x="14.6" y="8.3" width="1.9" height="9.6" rx="0.95" />
      {/* base flaring out to the plinth */}
      <path d="M7.1 18.4h9.8l1.5 1.9H5.6z" />
      <rect x="2.6" y="20.7" width="18.8" height="2.5" rx="1" />
    </svg>
  );
}
