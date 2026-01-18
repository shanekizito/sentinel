import { motion } from "framer-motion";
import { cn } from "@/lib/utils";

interface FloatingOrbProps {
  className?: string;
  color?: "primary" | "accent" | "violet" | "rose";
  size?: "sm" | "md" | "lg" | "xl";
  blur?: "sm" | "md" | "lg" | "xl";
  delay?: number;
  duration?: number;
}

const FloatingOrb = ({
  className,
  color = "primary",
  size = "md",
  blur = "lg",
  delay = 0,
  duration = 8,
}: FloatingOrbProps) => {
  const colors = {
    primary: "bg-primary/20",
    accent: "bg-accent/20",
    violet: "bg-violet/20",
    rose: "bg-rose/20",
  };

  const sizes = {
    sm: "w-[200px] h-[200px]",
    md: "w-[400px] h-[400px]",
    lg: "w-[600px] h-[600px]",
    xl: "w-[800px] h-[800px]",
  };

  const blurs = {
    sm: "blur-[50px]",
    md: "blur-[80px]",
    lg: "blur-[120px]",
    xl: "blur-[160px]",
  };

  return (
    <motion.div
      className={cn(
        "absolute rounded-full pointer-events-none",
        colors[color],
        sizes[size],
        blurs[blur],
        className
      )}
      animate={{
        y: [0, -30, 0],
        x: [0, 15, 0],
        scale: [1, 1.1, 1],
        opacity: [0.5, 0.8, 0.5],
      }}
      transition={{
        duration,
        repeat: Infinity,
        repeatType: "reverse",
        delay,
        ease: "easeInOut",
      }}
    />
  );
};

export { FloatingOrb };
