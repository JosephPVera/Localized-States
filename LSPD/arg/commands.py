# Written by Joseph P.Vera
# 2024-11

import argparse

class CommandLineArgs:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Generate localized defect plots.")
        
        # Basic plotting options
        self.parser.add_argument('--tot', action='store_true', help="Use the 'tot' mode for plotting")
        self.args = self.parser.parse_args()


    @property
    def tot_mode(self):
        return self.args.tot
    

    
