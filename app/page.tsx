export default function Home() {
  const year = new Date().getFullYear();

  return (
    <main className="container">
      <header className="hero">
        <h1>Niko Banto</h1>
        <p className="tagline">a senior at UH Manoa studying business management</p>
      </header>

      <section>
        <h2>This semester</h2>
        <ul>
          <li>Finishing my last year at Shidler</li>
          <li>Working at my job</li>
          <li>Surfing the North Shore once the season starts</li>
        </ul>
      </section>

      <section>
        <h2>About</h2>
        <p>
          I&apos;m a senior at UH Manoa, where I study business management. My
          coursework focuses on how organizations are run, led and grown. As I
          finish my degree, I&apos;m working to put what I&apos;ve learned into
          practice.
        </p>
      </section>

      <footer className="footer">
        <p>© {year} Niko Banto · Built with Claude Code</p>
      </footer>
    </main>
  );
}
