import matplotlib
matplotlib.use('Agg')

import sys; sys.path += ['../compiler']
import shutil
import os
import glob
import skimage, skimage.io, skimage.color
import subprocess
import numpy
from render_metrics import *
import json
import tuner
import dill
import random

import matplotlib.pyplot as plt
import pylab

try:
    input_cmd = raw_input
except:
    input_cmd = input

def main():
    args = sys.argv[1:]
    if len(args) < 2:
        print('python collect_webpage.py basedir out_webdir [description] [--no-predict] [--subdir] [--video] [--heatmap]')
        print('  Here basedir is the out directory given to render_all.py/learn_all.py and')
        print('  out_webdir is the output directory to place the website in.')
        print('  description is a quoted text with an additional human-readable description of the experiment')
        print('  --no-predict disables the prediction column (neural network learning)')
        print('  --subdir indicates that basedir is actually the output of a given shader (a subdirectory of the base directory)')
        print('  --video indicates that a video should be built using ffmpeg and linked from the webpage')
        print('  --heatmap indicates to show an inset heat-map comparing to ground truth')
        sys.exit(1)

    basedir = args[0]
    webdir = args[1]
    description = ''
    use_predict = not '--no-predict' in args
    if len(args) > 2:
        description = args[2]
    video = False
    if '--video' in args:
        video = True
    no_ground_truth = '--no-ground-truth' in args
    heatmap = '--heatmap' in args
    
    if os.path.exists(webdir):
        shutil.rmtree(webdir)
    if not os.path.exists(webdir):
        os.makedirs(webdir)

    def read_or_empty(fname):
        if os.path.exists(fname):
            return open(fname, 'rt').read()
        return ''

    def get_video_filename(suffix):
        return 'video' + suffix + '.mp4'

    with open(os.path.join(webdir, 'index.html'), 'wt') as f:
        f.write('<html><body>\n')
        git_version = subprocess.check_output('git rev-parse HEAD', shell=True).strip()
        #video_filename = 'video.mp4'
        f.write('<h1>Shader Results</h1>')
        f.write('<h3>For git commit %s</h3>\n' % git_version.decode('utf-8'))
        if video:
            f.write('<h3>Videos: ')
            if not no_ground_truth:
                f.write('<a href="%s">Ground truth</a>'%get_video_filename('ground'))
            f.write('<a href="%s">f</a>' % get_video_filename('f'))
            f.write('<a href="%s">g</a></h3>' % (get_video_filename('g')))
            
            for prefix in ['ground', 'f', 'g']:
                prefix_name = prefix
                if prefix_name == 'ground':
                    prefix_name = 'Ground truth'
                if no_ground_truth and prefix == 'ground':
                    continue
                f.write(prefix_name + ':<br><video controls><source src="%s" type="video/mp4"></video>' % get_video_filename(prefix))
                f.write('<br>')
        f.write('<h3>%s</h3>\n' % description)
        f.write('<table border=2 cellpadding=2 cellspacing=0>\n')
        f.write('<tr><td><b>Name (and link to info)</b></td><td><b>Ground truth</b></td><td><b>1 sample (f)</b></td><td><b>Bandlimited (g)</b></td><td><b>Learned result</b></td></tr>')
        metrics_all = {}
        metrics_keys = ['L2', 'highfreq']
        if not '--subdir' in args:
            subdirL = sorted(glob.glob(os.path.join(basedir, '*')))
        else:
            subdirL = [basedir]
        for subdir in subdirL:
            if os.path.isdir(subdir):
                #print(subdir)
                subdir_name = os.path.split(subdir)[1]

                all_file = os.path.join(subdir, 'tune', 'all.dill')
                if os.path.exists(all_file):
                    all_pop = dill.loads(open(all_file, 'rb').read())[tuner.ours_label]
                else:
                    all_pop = None

                filenameL = sorted(glob.glob(os.path.join(subdir, 'ground*.png')))
                for filename in filenameL:
                
                    print('Processing', subdir, filename)

                    filename_strip = os.path.split(filename)[1][len('ground'):]
                    filename_index = int(os.path.splitext(filename_strip)[0])

                    info_name = subdir_name + '%05d.txt' % filename_index
                    full_info_name = os.path.join(webdir, info_name)
                    if not video:
                        f.write('<tr>')
                        f.write('<td><a href="%s">%s</a></td>'%(info_name, subdir_name))
                    
                    indiv = None
                    with open(full_info_name, 'wt') as info_f:
                        info_f.write('Render command:\n' + read_or_empty(os.path.join(subdir, 'render_command.txt')).strip() + '\n\n')
                        info_f.write('Render time:\n' + read_or_empty(os.path.join(subdir, 'render_time.txt')).strip() + '\n\n')
                        info_f.write('Learn command:\n' + read_or_empty(os.path.join(subdir, 'learn_command.txt')).strip() + '\n\n')
                        info_f.write('Learn time:\n' + read_or_empty(os.path.join(subdir, 'learn_time.txt')).strip() + '\n\n')
                        pareto_idx = read_or_empty(os.path.join(subdir, 'pareto_index%05d.txt'%filename_index)).strip()
                        info_f.write('Pareto index:\n' + pareto_idx + '\n\n')
                        if all_pop is not None:
                            try:
                                pareto_idx = int(pareto_idx)
                            except:
                                pareto_idx = None
                            if pareto_idx is not None and pareto_idx < len(all_pop):
                                indiv = all_pop[pareto_idx]
                                info_f.write('Individual:\n\n' + repr(all_pop[pareto_idx]) + '\n\n')
                
                    metrics_row = {}
                    
                    for prefix in ['ground', 'f', 'g'] + (['predict'] if use_predict else []):
                        #print(prefix, filename_strip)
                        filename_prefix = prefix + filename_strip
                        full_filename_prefix = os.path.join(subdir, filename_prefix)
                        dest_filename = subdir_name + '_' + os.path.split(filename_prefix)[1]
                        full_dest_filename = os.path.join(webdir, dest_filename)
                        render_info = read_or_empty(os.path.join(subdir, 'render_info%05d.json'%filename_index))
                        if render_info == '':
                            render_info = {}
                        else:
                            render_info = json.loads(render_info)
                        #print(filename, filename_strip, render_info)
                        #print(full_filename_prefix, full_dest_filename)
                        
                        if os.path.exists(full_filename_prefix):
                            shutil.copyfile(full_filename_prefix, full_dest_filename)

                        heatmap_filename = None
                        if heatmap and prefix in ['f', 'g']:
                            I0 = skimage.img_as_float(skimage.io.imread(full_dest_filename))
                            I_diff = (ground_I - I0)**2
                            if len(I_diff.shape) == 3:
                                I_diff = numpy.sum(I_diff, 2)
                            I_diff = numpy.sqrt(I_diff)
                            #pylab.clf()
                            #
                            #pylab.imshow(I_diff, cmap='inferno')
                            heatmap_filename = os.path.join(webdir, os.path.splitext(dest_filename)[0] + '_heatmap.png')
                            #pylab.savefig(heatmap_filename)
                            fig = plt.figure(frameon=False)
                            fig.set_size_inches(1.0,2.0/3.0)

                            ax = plt.Axes(fig, [0., 0., 1., 1.])
                            ax.set_axis_off()
                            fig.add_axes(ax)
                            if len(I0.shape) == 2:
                                I0 = numpy.dstack((I0,)*3)
#                            # Testing code to check heatmap scales
#                            if random.random() < 0.5:
#                                I_diff = numpy.random.random(I_diff.shape)
#                            else:
#                                I_diff = numpy.random.random(I_diff.shape)*0.25
                            
                            ax.imshow(I0, aspect='normal')

                            subsize = 0.3
                            ax = plt.Axes(fig, [1-subsize, 0.0, subsize, subsize])
                            ax.set_axis_off()
                            fig.add_axes(ax)
                            ax.imshow(I_diff, aspect='normal', cmap='hot', vmin=0.0, vmax=1.0)
                            
                            fig.savefig(heatmap_filename, dpi=480*2)
                        
                        def report_metrics():
                            predict_I = skimage.img_as_float(skimage.io.imread(full_dest_filename))
                            if (len(predict_I.shape) == 2 or (len(predict_I.shape) == 3 and predict_I.shape[2] == 1)) and (len(ground_I.shape) == 3 and ground_I.shape[2] == 3):
                                predict_I = skimage.color.gray2rgb(predict_I)
                            L2 = metric_L2(ground_I, predict_I)
                            highfreq_L2 = metric_highfreq(ground_I, predict_I)
                            return ({'L2': L2, 'highfreq': highfreq_L2}, '<i>L</i><sup>2</sup> error: %.6f<br>High freq error: %.6f' % (L2, highfreq_L2))

                        below = ''
                        if prefix == 'ground':
                            ground_I  = skimage.img_as_float(skimage.io.imread(full_dest_filename))
                            resolution = ground_I.shape[0] * ground_I.shape[1]
                            for key in metrics_keys:
                                metrics_row.setdefault(key, []).append(0.0)
                        elif prefix in ['f', 'g', 'predict']:
                            (metrics, metrics_str) = report_metrics()
                            
                            #print('collecting, pareto_idx=%r, error image: %f' % (pareto_idx, metrics['L2']))
                            if pareto_idx is not None and all_pop is not None and pareto_idx < len(all_pop):
                                setattr(all_pop[pareto_idx], 'error_' + prefix + '_image', metrics['L2'])
                                #print([hasattr(p, 'error_g_image') for p in all_pop])
                            
                            below = '<br>' + metrics_str
                            for key in metrics_keys:
                                metrics_row.setdefault(key, []).append(metrics[key])
                            #f.write('<td>%.6f</td>' % L2)

                        if prefix == 'predict':
                            sparsity = read_or_empty(os.path.join(subdir, 'model_sparsity.txt'))
                            if len(sparsity):
                                sparsity = '%.3f' % float(sparsity)
                                below += '<br>Sparsity: %s' % sparsity
                        
                        if 'time_f' in render_info and prefix == 'f':
                            t = render_info['time_f']
                            below += '<br>Time: %.3f ns/pixel<br>Time: %.3f ms/frame' % (t*1e9, t*resolution*1e3)
                        if 'time_g' in render_info and prefix == 'g':
                            t = render_info['time_g']
                            below += '<br>Time: %.3f ns/pixel<br>Time: %.3f ms/frame' % (t*1e9, t*resolution*1e3)
                        if indiv is not None and hasattr(indiv, 'error_f') and prefix == 'f':
                            below += '<br>Error (L2, command-line): %s' % (indiv.error_f)
                        if indiv is not None and hasattr(indiv, 'error_g') and prefix == 'g':
                            below += '<br>Error (L2, command-line): %s' % (indiv.error_g)
                        
                        if not video:
                            f.write('<td valign=top><center><img src="%s">%s</center></td>' % (os.path.split(heatmap_filename)[1] if heatmap_filename is not None else dest_filename, below))
                    for key in metrics_keys:
                        metrics_all.setdefault(key, []).append(metrics_row[key])
                    if not video:
                        f.write('</tr>\n')

                    if all_pop is not None:
                        all_image_file = os.path.join(subdir, 'tune', 'all_image.dill')
                        with open(all_image_file, 'wb') as f_image:
                            f_image.write(dill.dumps({tuner.ours_label: all_pop}))

                if video:
                    #ffmpeg -r 60 -f image2 -s 1920x1080 -i pic%04d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p test.mp4
                    #print(webdir)
                    #print(subdir)
                    def encode_video(suffix):
                        cmd = 'ffmpeg -i %s -r 30 -c:v libx264 -preset slow -crf 0 %s' % (os.path.join(webdir, subdir_name + '_' + suffix + '%10d.png'), os.path.join(webdir, get_video_filename(suffix)))
                        print(cmd)
                        os.system(cmd)
                    
                    encode_video('ground')
                    encode_video('f')
                    encode_video('g')

        if len(subdirL):
            for key in metrics_keys:
                metrics_matrix = metrics_all[key]
                metrics_mean = numpy.mean(metrics_matrix, 0)
                for statistic in ['mean']:
                    f.write('<tr><td>%s %s</td>'%(statistic, key))
                    for col in range(len(metrics_matrix[0])):
                        f.write('<td>%.6f</td>' % metrics_mean[col])
                    f.write('</tr>')
        f.write('</table></body></html>\n')


if __name__ == '__main__':
    main()
