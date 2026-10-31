"""Execute analysis-only cells against completed real outputs, preserving earlier notebook cells."""
from pathlib import Path
import sys,base64,io
import nbformat
from IPython.utils.capture import capture_output
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];sys.path.insert(0,str(HERE))
import decoder_experiment as de
import pandas as pd
scope={'HERE':HERE,'ROOT':ROOT,'ex':de.ex,'pd':pd}
p=HERE/'decoder_tuning.ipynb';nb=nbformat.read(p,as_version=4);active=False;number=100
for cell in nb.cells:
    if cell.cell_type=='markdown' and cell.source.startswith('## A–C:'):active=True
    if not active or cell.cell_type!='code':continue
    number+=1;plt.close('all')
    with capture_output() as out:exec(compile(cell.source,str(p),'exec'),scope)
    cell.execution_count=number;cell.outputs=[]
    if out.stdout:cell.outputs.append(nbformat.v4.new_output('stream',name='stdout',text=out.stdout))
    if out.stderr:cell.outputs.append(nbformat.v4.new_output('stream',name='stderr',text=out.stderr))
    for item in out.outputs:cell.outputs.append(nbformat.v4.new_output('display_data',data=item.data,metadata=item.metadata))
    for f in plt.get_fignums():
        b=io.BytesIO();plt.figure(f).savefig(b,format='png',dpi=120,bbox_inches='tight');cell.outputs.append(nbformat.v4.new_output('display_data',data={'image/png':base64.b64encode(b.getvalue()).decode()},metadata={}))
nbformat.write(nb,p);print('Refreshed analysis cells',number-100)
