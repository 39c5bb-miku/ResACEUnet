import os
from tqdm import tqdm
import numpy as np
import segyio

def read_inregular_seisdata(file,total_trace=None):
	MAXCDP = 0
	MINCDP = 1000000
	MAXINLINE = 0
	MININLINE = 1000000
	with segyio.open(file, strict=True,ignore_geometry=True) as src:
		for trace_index in tqdm(range(src.tracecount)):
			trace_header = src.header[trace_index]
			INLINE_3D = trace_header[segyio.TraceField.INLINE_3D]
			if INLINE_3D==0:
				INLINE_3D = trace_header[segyio.TraceField.FieldRecord]
			CDP = trace_header[segyio.TraceField.CDP]
			if CDP==0:
				CDP = trace_header[segyio.TraceField.CROSSLINE_3D]
			MAXINLINE=max(MAXINLINE,INLINE_3D)
			MAXCDP = max(MAXCDP, CDP)
			MININLINE=min(MININLINE,INLINE_3D)
			MINCDP = min(MINCDP, CDP)
		data = np.asarray([np.copy(x) for x in src.trace[:]])
		n1, n2 = data.shape
		total=np.zeros((MAXINLINE-MININLINE+1,MAXCDP-MINCDP+1,n2),dtype=np.float32)
		for trace_index in tqdm(range(n1)):
			trace_header = src.header[trace_index]
			INLINE_3D = trace_header[segyio.TraceField.INLINE_3D]
			if INLINE_3D==0:
				INLINE_3D = trace_header[segyio.TraceField.FieldRecord]
			CDP = trace_header[segyio.TraceField.CDP]
			if CDP==0:
				CDP = trace_header[segyio.TraceField.CROSSLINE_3D]
			if total_trace is not None:
				total[INLINE_3D - MININLINE, CDP - MINCDP, :] = total_trace[trace_index]
			else:
				total[INLINE_3D-MININLINE,CDP-MINCDP,:]=data[trace_index]
	return total,n1


def create_segyfile_by_npy(begin_inline,begin_crossline,begin_time,
				time_interval, inputfilename,inc_inline=1,inc_crossline=1,inline=189,crossline=193):

	filename_decode = inputfilename.split('.')
	if len(filename_decode)>2:
		prefix = '.'.join(filename_decode[:-1])
	else:
		prefix = filename_decode[0]
	data = np.load(inputfilename)
	n1,n2,n3=data.shape
	end_inline = begin_inline + n1 - 1
	end_crossline = begin_crossline + n2 - 1
	end_time = begin_time + time_interval * (n3 - 1)
	totaltrace=(end_inline-begin_inline+1)*(end_crossline-begin_crossline+1)
	data = data.reshape(totaltrace,n3)
	spec=segyio.spec()
	spec.iline = inline
	spec.xline = crossline
	spec.ilines=[i for i in range(begin_inline,end_inline+inc_inline,inc_inline)]
	spec.xlines = [i for i in range(begin_crossline, end_crossline+inc_crossline, inc_crossline)]
	spec.samples  = [i for i in range(begin_time, end_time+time_interval, time_interval)]
	spec.sorting = 2
	spec.format = 1
	with segyio.create(prefix+'.sgy', spec) as dst:
		for trace_index in tqdm(range(dst.tracecount)):
			dst.header[trace_index][segyio.TraceField.TRACE_SEQUENCE_LINE]=trace_index+1
			dst.header[trace_index][segyio.TraceField.INLINE_3D]=(trace_index)//int((end_crossline-begin_crossline)//inc_crossline+1)*inc_inline + begin_inline
			dst.header[trace_index][segyio.TraceField.CROSSLINE_3D]=(trace_index)%int((end_crossline-begin_crossline)//inc_crossline+1)*inc_crossline + begin_crossline
			dst.header[trace_index][segyio.TraceField.DelayRecordingTime] = begin_time
			dst.header[trace_index][segyio.TraceField.LagTimeA] = begin_time
			dst.header[trace_index][segyio.TraceField.MuteTimeStart] = begin_time
			dst.header[trace_index][segyio.TraceField.MuteTimeEND] = end_time
			dst.header[trace_index][segyio.TraceField.TRACE_SAMPLE_COUNT]= (end_time-begin_time)//time_interval+1
			dst.header[trace_index][segyio.TraceField.TRACE_SAMPLE_INTERVAL] = time_interval
			dst.header[trace_index][segyio.TraceField.CDP_X] = (trace_index)//int((end_crossline-begin_crossline)//inc_crossline+1)*20 + 1245788
			dst.header[trace_index][segyio.TraceField.CDP_Y] = (trace_index)%int((end_crossline-begin_crossline)//inc_crossline+1)*20 + 1245577
		dst.trace = data
	dst.close()


def create_segyfile_by_bin(begin_inline,end_inline,begin_crossline,end_crossline,begin_time,end_time,
				time_interval, inputfilename,inc_inline=1,inc_crossline=1,inline=189,crossline=193):

	totaltrace=(end_inline-begin_inline+1)*(end_crossline-begin_crossline+1)
	filename_decode = inputfilename.split('.')
	if len(filename_decode)>2:
		prefix = '.'.join(filename_decode[:-1])
	else:
		prefix = filename_decode[0]
	data = np.fromfile(inputfilename,dtype=np.float32)
	n1=data.shape[0]
	data = data.reshape(totaltrace,n1//totaltrace)
	spec=segyio.spec()
	spec.iline = inline
	spec.xline = crossline
	spec.ilines=[i for i in range(begin_inline,end_inline+inc_inline,inc_inline)]
	spec.xlines = [i for i in range(begin_crossline, end_crossline+inc_crossline, inc_crossline)]
	spec.samples  = [i for i in range(begin_time, end_time+time_interval, time_interval)]
	spec.sorting = 2
	spec.format = 1
	with segyio.create(prefix+'.sgy', spec) as dst:
		for trace_index in tqdm(range(dst.tracecount)):
			dst.header[trace_index][segyio.TraceField.TRACE_SEQUENCE_LINE]=trace_index+1
			dst.header[trace_index][segyio.TraceField.INLINE_3D]=(trace_index)//int((end_crossline-begin_crossline)//inc_crossline+1)*inc_inline + begin_inline
			dst.header[trace_index][segyio.TraceField.CROSSLINE_3D]=(trace_index)%int((end_crossline-begin_crossline)//inc_crossline+1)*inc_crossline + begin_crossline
			dst.header[trace_index][segyio.TraceField.DelayRecordingTime] = begin_time
			dst.header[trace_index][segyio.TraceField.LagTimeA] = begin_time
			dst.header[trace_index][segyio.TraceField.MuteTimeStart] = begin_time
			dst.header[trace_index][segyio.TraceField.MuteTimeEND] = end_time
			dst.header[trace_index][segyio.TraceField.TRACE_SAMPLE_COUNT]= (end_time-begin_time)//time_interval+1
			dst.header[trace_index][segyio.TraceField.TRACE_SAMPLE_INTERVAL] = time_interval
			dst.header[trace_index][segyio.TraceField.CDP_X] = (trace_index)//int((end_crossline-begin_crossline)//inc_crossline+1)*20 + 1245788
			dst.header[trace_index][segyio.TraceField.CDP_Y] = (trace_index)%int((end_crossline-begin_crossline)//inc_crossline+1)*20 + 1245577
		dst.trace = data
	dst.close()


def save_inregular_seisdata(temp,input_folder,output_folder,file,fix='ai'):
	filename, subfix = file.split('.')
	n1,n2,n3=temp.shape
	MAXCDP = 0
	MINCDP = 1000000
	MAXINLINE = 0
	MININLINE = 1000000
	with segyio.open(input_folder + '/' + filename + '.' + subfix, strict=True, ignore_geometry=True) as src:
		for trace_index in tqdm(range(src.tracecount)):
			trace_header = src.header[trace_index]
			INLINE_3D = trace_header[segyio.TraceField.INLINE_3D]
			if INLINE_3D==0:
				INLINE_3D = trace_header[segyio.TraceField.FieldRecord]
			CDP = trace_header[segyio.TraceField.CDP]
			if CDP==0:
				CDP = trace_header[segyio.TraceField.CROSSLINE_3D]
			MAXINLINE=max(MAXINLINE,INLINE_3D)
			MAXCDP = max(MAXCDP, CDP)
			MININLINE=min(MININLINE,INLINE_3D)
			MINCDP = min(MINCDP, CDP)
		data=np.zeros((n1*n2,n3),dtype=np.float32)
		for trace_index in tqdm(range(src.tracecount)):
			trace_header = src.header[trace_index]
			INLINE_3D = trace_header[segyio.TraceField.INLINE_3D]
			if INLINE_3D == 0:
				INLINE_3D = trace_header[segyio.TraceField.FieldRecord]
			CDP = trace_header[segyio.TraceField.CDP]
			if CDP == 0:
				CDP = trace_header[segyio.TraceField.CROSSLINE_3D]
			data[trace_index]= temp[INLINE_3D - MININLINE, CDP - MINCDP, :]

		spec = segyio.tools.metadata(src)
		with segyio.create(output_folder + '/' + filename + f'_{fix}.' + subfix, spec) as dst:
			dst.text[0] = src.text[0]
			dst.bin = src.bin
			dst.header = src.header
			dst.trace = data
		dst.close()

def dat_npy(dat_file,npy_file,shape=(128,128,128)):
    with open(dat_file, 'r') as f:
        data = np.fromfile(f, dtype=np.float32)
        data = data.reshape(shape)

    np.save(npy_file, data)

if __name__ == '__main__':
	output_folder=r't'
	input_folder=r''
	sgy=r''
	npy=r''
	# results = np.load(os.path.join(output_folder,npy))
	# save_inregular_seisdata(results,input_folder,output_folder,sgy,fix='ai')
	create_segyfile_by_npy(1,1,0,1,npy)
	# dat_npy(output_folder,input_folder,shape=(128,128))
