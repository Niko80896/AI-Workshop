export default function Home() {
  const year = new Date().getFullYear();

  return (
    <main className="container">
      <header className="hero">
        <h1>Niko Banto</h1>
        <p className="tagline">a senior at UH Manoa studying business management</p>
      </header>

      <section>
        <h2>About</h2>
        <p>
          I&apos;m a senior at UH Manoa, where I study business management. My
          coursework focuses on how organizations are run, led and grown. As I
          finish my degree, I&apos;m working to put what I&apos;ve learned into
          practice.
        </p>
      </section>

      <section>
        <h2>This semester</h2>
        <ul>
          <li>Completing my strategic management capstone with a full analysis of a local Hawaii business.</li>
          <li>Leading a four-person team project on improving operations for a campus organization.</li>
          <li>Polishing my resume and practicing case interviews ahead of spring recruiting.</li>
        </ul>
      </section>

      <footer className="footer">
        <p>© {year} Niko Banto</p>
      </footer>
    </main>
  );
}
