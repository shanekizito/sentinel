import * as React from "react";
import { motion, HTMLMotionProps } from "framer-motion";
import { cn } from "@/lib/utils";

interface PremiumButtonProps extends Omit<HTMLMotionProps<"button">, "children"> {
  children: React.ReactNode;
  variant?: "primary" | "secondary" | "outline" | "ghost" | "cyber" | "glass";
  size?: "sm" | "md" | "lg" | "xl";
  glow?: boolean;
  icon?: React.ReactNode;
  iconPosition?: "left" | "right";
}

const PremiumButton = React.forwardRef<HTMLButtonElement, PremiumButtonProps>(
  ({ 
    className, 
    children, 
    variant = "primary", 
    size = "md",
    glow = false,
    icon,
    iconPosition = "right",
    ...props 
  }, ref) => {
    const baseStyles = "relative inline-flex items-center justify-center gap-2 font-medium rounded-full transition-all duration-300 overflow-hidden group disabled:opacity-50 disabled:pointer-events-none";
    
    const variants = {
      primary: "bg-primary text-primary-foreground hover:bg-primary/90",
      secondary: "bg-secondary text-secondary-foreground hover:bg-secondary/80",
      outline: "border-2 border-primary/30 text-foreground hover:border-primary hover:bg-primary/10",
      ghost: "text-foreground hover:bg-white/5",
      cyber: "bg-gradient-to-r from-primary via-accent to-primary bg-[length:200%_100%] text-primary-foreground animate-gradient",
      glass: "backdrop-blur-xl bg-white/10 border border-white/20 text-foreground hover:bg-white/15",
    };
    
    const sizes = {
      sm: "h-9 px-4 text-sm",
      md: "h-11 px-6 text-sm",
      lg: "h-13 px-8 text-base",
      xl: "h-14 px-10 text-base",
    };
    
    const glowStyles = glow ? "shadow-lg shadow-primary/30 hover:shadow-xl hover:shadow-primary/40" : "";
    
    return (
      <motion.button
        ref={ref}
        className={cn(baseStyles, variants[variant], sizes[size], glowStyles, className)}
        whileHover={{ scale: 1.02, y: -2 }}
        whileTap={{ scale: 0.98 }}
        transition={{ type: "spring", stiffness: 400, damping: 17 }}
        {...props}
      >
        {/* Shimmer effect */}
        <span className="absolute inset-0 overflow-hidden rounded-full">
          <span className="absolute inset-0 translate-x-[-100%] bg-gradient-to-r from-transparent via-white/20 to-transparent group-hover:animate-shimmer" />
        </span>
        
        {/* Border glow animation for cyber variant */}
        {variant === "cyber" && (
          <span className="absolute inset-0 rounded-full">
            <span className="absolute inset-[-2px] rounded-full bg-gradient-to-r from-primary via-accent to-primary opacity-0 group-hover:opacity-100 blur-sm transition-opacity" />
          </span>
        )}
        
        {/* Content */}
        <span className="relative z-10 flex items-center gap-2">
          {icon && iconPosition === "left" && (
            <span className="transition-transform group-hover:-translate-x-0.5">{icon}</span>
          )}
          {children}
          {icon && iconPosition === "right" && (
            <span className="transition-transform group-hover:translate-x-0.5">{icon}</span>
          )}
        </span>
      </motion.button>
    );
  }
);

PremiumButton.displayName = "PremiumButton";

export { PremiumButton };
