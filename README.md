# Microbial-Mutual-Information
This record includes software and data used to compute shared information between streamwater microbial taxa and hydrologic metrics for the article:  
URycki, D. R., Bassiouni, M., Good, S. P., Crump, B. C., &amp; Li, B. (2022). The streamwater microbiome encodes hydrologic data across scales. Science of The Total Environment, 157911. https://doi.org/10.1016/j.scitotenv.2022.157911

To run this analysis:
1. Download all files and folders into a primary parent directory.
2. Load necessary Python libraries using the file 'requirements.txt' -OR- 
	create a virtual environment with the file 'genohydro_mi.yml.' 
3. First run the script '01_compute_microbial_MI.py'
4. After the first script has completed, run the second script '02_analyze_microbial_MI.py'


Notes: 
a. These scripts were developed on Windows OS, but might run on Linux-like systems.
b. The second script depends on output files from the first script.
c. The script 'genohydro.py' contains helper functions necessary for the first script to run.

Please contact the corresponding author for questions, assistance, or comments: dawn.urycki@ucdenver.edu
