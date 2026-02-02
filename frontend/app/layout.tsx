import type { Metadata } from "next";
import { Inter, Outfit, Instrument_Serif } from "next/font/google";
import "./globals.css";
import { Toaster } from "react-hot-toast";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
});

const outfit = Outfit({
  variable: "--font-outfit",
  subsets: ["latin"],
});

const instrumentSerif = Instrument_Serif({
  variable: "--font-serif",
  weight: "400",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "ResMe",
  description: "Revive your career.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body
        className={`${inter.variable} ${outfit.variable} ${instrumentSerif.variable} antialiased bg-black text-white`}
      >
        <Toaster
          position="top-right"
          toastOptions={{
            style: {
              background: "rgba(15, 23, 42, 0.8)",
              color: "#fff",
              border: "1px solid rgba(255, 255, 255, 0.1)",
              backdropFilter: "blur(8px)",
            },
          }}
        />
        {children}
      </body>
    </html>
  );
}
