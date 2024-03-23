from glob import glob
from numpy import unique
os.environ['OPENBLAS_NUM_THREADS'] = '1' #needed for slurm numpy executions

nd2 = glob('{}/*'.format(config['nd2Dir'])) #gets unique filesnames
samples = []
for i in nd2:
  fileName = i.replace('{}/'.format(config['nd2Dir']), '') #format adds the files into {}.
  fileName = fileName.replace('{}'.format(config['filesuff']), '')
  samples.append(fileName)
samples = unique(samples)
print(samples)

rule all:
  input:
    expand(config['workDir'] +'tifs/{sample}.ome.tiff', sample=samples)

rule nd2tiff:
  input:
    nd2 = config['nd2Dir'] + '/{sample}' + config['filesuff'],
  output:
    tiff = config['workDir'] +'tifs/{sample}.ome.tiff',
  shell:
    'set +u && '
    'module purge && '
    'eval "$(conda shell.bash hook)" && '
    'conda activate bfconvert && '
    'BF_MAX_MEM=24g bfconvert {input.nd2} {output.tiff}'

rule getCSVtiff:
  input:
    tiff = config['workDir'] +'tifs/{sample}.tif',
  output:
    brightest_frame = config['workDir'] + 'figures/{sample}__brightestframe_contrasted.pdf',
    lanes_cleaned = config['workDir'] + 'figures/{sample}_lanes_detected.pdf',
    csv = config['workDir'] + 'csv/{sample}.csv',
  params:
    getcsv = config['pathtocsvpy']
  shell:
    'python {params.getcsv} {input.tiff} {output.brightest_frame} {output.lanes_cleaned} {output.csv}'

rule multilineage:
  input:
    csv = config['workDir'] + 'csv/{sample}.csv',
  output:
    plots = config['workDir'] + 'plots/',
    filtered = config['workDir'] + 'filtered_traces/',
    datacsv = config['workDir'] + 'stats.xlxs'
