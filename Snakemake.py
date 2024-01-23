from glob import glob
from numpy import unique

nd2 = glob('{}/*'.format(config['nd2Dir']))#gets unique filesnames
samples = []
for i in nd2:
  fileName = i.replace('{}/'.format(config['nd2Dir']), '') #format adds the files into {}.
  fileName = fileName.replace('{}'.format(config['filesuff']), '')
  samples.append(fileName)
samples = unique(samples)
print(samples)

rule all:
  input:
    expand('csv/{sample}.csv', file=files)

rule nd2tiff:
  input:
    nd2 = config['nd2Dir'] + '/{sample}' + config['filesuff'],
  output:
    tiff = 'tifs/{sample}.tif',
  params:
    bftools = config['pathtobftools']
  shell:
    'BF_MAX_MEM=2G {params.bftools}/bfconvert -no-upgrade -bigtiff -compression zlib {input.nd2} {output.tiff}'

rule getCSVtiff:
  input:
    tiff = 'tifs/{sample}.tif',
  output:
    csv = 'csv/{sample}.csv',
  params:
    getcsv = config['pathtocsvpy']
  shell:
    'python {params.getcsv} {input.tiff} {output.csv}'
