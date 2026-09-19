import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Gesture Control Lab · Ten-Gesture ONNX',
  description: 'Ten-FPS dual-camera gesture control with ten commands, 3-D palm angles, corrected Left/Right semantics, open-set rejection, stabilized landmarks, and validation-gated feedback learning.',
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
