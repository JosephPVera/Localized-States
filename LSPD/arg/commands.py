# Written by Joseph P.Vera
# 2025-02

class CommandLineArgs:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Generate localized defect plots.")
        
        # Basic plotting options
        self.parser.add_argument('--tot', action='store_true', help="Use the 'tot' mode for plotting")
        self.parser.add_argument('--band', action='store_true', help="Display band numbers on the plot")
        self.parser.add_argument('--gamma', action='store_true', help="only for gamma calculations")
        self.parser.add_argument('--split', action='store_true', help="split the degenerate states")
        self.args = self.parser.parse_args()

    @property
    def tot_mode(self):
        return self.args.tot

    @property
    def band_mode(self):
        return self.args.band
        
    @property
    def gamma(self):
        return self.args.gamma

    @property
    def split_mode(self):
        return self.args.split  
