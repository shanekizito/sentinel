import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import { Menu, X, ArrowRight } from "lucide-react";
import { useState } from "react";

const Navbar = () => {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const navLinks = [
    { label: "Home", href: "/" },
    { label: "How It Works", href: "/how-it-works" },
    { label: "Security", href: "/security" },
    { label: "Automation", href: "/automation" },
    { label: "Docs", href: "/docs" },
    { label: "Live Engine", href: "/engine" },
    { label: "Contact", href: "/contact" },
  ];

  return (
    <motion.nav
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: "easeOut" }}
      className="fixed top-0 left-0 right-0 z-50"
    >
      <div className="absolute inset-0 bg-background/80 backdrop-blur-md border-b border-line-muted" />

      <div className="container mx-auto px-6 relative">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <Link to="/" className="flex items-center gap-1 group">
            {/* Custom Sentinel Logo */}
            <img src="/logo.png" alt="Sentinel Logo" className="w-[60px] h-auto object-contain" />
            <span className="font-display font-bold text-2xl text-foreground tracking-tight">
              Sentinel
            </span>
          </Link>

          {/* Navigation Links - Desktop */}
          <div className="hidden lg:flex items-center">
            <div className="flex items-center border border-line-muted rounded-full px-1 py-1 bg-background/50">
              {navLinks.map((link, index) => (
                <Link
                  key={link.href + index}
                  to={link.href}
                  className="px-4 py-2 text-sm text-muted-foreground hover:text-foreground transition-colors rounded-full hover:bg-card/80"
                >
                  {link.label}
                </Link>
              ))}
            </div>
          </div>

          {/* CTA Button */}
          <div className="hidden lg:flex items-center gap-4">
            <button className="btn-primary">
              Run Scan
              <ArrowRight className="w-4 h-4" />
            </button>
          </div>

          {/* Mobile menu button */}
          <button
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="lg:hidden w-10 h-10 rounded-lg flex items-center justify-center border border-line-muted hover:bg-card transition-colors"
          >
            {mobileMenuOpen ? (
              <X className="w-5 h-5 text-foreground" />
            ) : (
              <Menu className="w-5 h-5 text-foreground" />
            )}
          </button>
        </div>
      </div>

      {/* Mobile menu */}
      {mobileMenuOpen && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="lg:hidden absolute top-full left-0 right-0 bg-background/98 backdrop-blur-xl border-b border-line-muted"
        >
          <div className="container mx-auto px-6 py-4 space-y-1">
            {navLinks.map((link, index) => (
              <Link
                key={link.href + index}
                to={link.href}
                onClick={() => setMobileMenuOpen(false)}
                className="block px-4 py-3 rounded-lg text-base text-foreground hover:bg-card transition-colors"
              >
                {link.label}
              </Link>
            ))}
            <div className="pt-4 border-t border-line-muted mt-4">
              <button className="btn-primary w-full justify-center">
                Run Scan
                <ArrowRight className="w-4 h-4" />
              </button>
            </div>
          </div>
        </motion.div>
      )}
    </motion.nav>
  );
};

export default Navbar;