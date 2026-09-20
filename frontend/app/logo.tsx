/* eslint-disable @next/next/no-img-element */
export function Logo({ size = 20 }: { size?: number }) {
  return (
    <img
      src="/logo.png"
      alt=""
      width={size}
      height={size}
      style={{ display: "block", width: size, height: size }}
    />
  );
}
