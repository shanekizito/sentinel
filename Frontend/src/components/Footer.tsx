import { Link } from "react-router-dom";

const Footer = () => {
  return (
    <footer className="py-12 border-t border-line-muted bg-background">
      <div className="container mx-auto px-6">
        <div className="flex flex-col md:flex-row justify-between items-center gap-8">

          {/* Left */}
          <div className="text-center md:text-left">
            <h3 className="font-display font-bold text-xl mb-1">Sentinel</h3>
            <p className="text-sm text-muted-foreground">Security for AI-Written Software</p>
          </div>

          {/* Right Links */}
          <div className="flex flex-wrap justify-center gap-x-8 gap-y-2">
            {["Documentation", "Security Model", "Automation Guides", "Roadmap", "Contact"].map((link) => (
              <a key={link} href="#" className="text-sm text-muted-foreground hover:text-foreground transition-colors">
                {link}
              </a>
            ))}
          </div>
        </div>

        <div className="mt-12 pt-8 border-t border-line-muted flex flex-col md:flex-row justify-between items-center gap-4 text-xs text-muted-foreground">
          <p>© 2024 Sentinel. All rights reserved.</p>
          <p>Designed for Trust.</p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;