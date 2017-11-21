# mfcc
Calculate MFCC/Fbank feature for wav files

## Install and Usage

Support python **3.6** only!

To use, make sure you have install **SCIPY** lib then import **MFCC** modual by:

<pre>
import mfcc
</pre>

Simply run the following command for MFCC feature:
<pre>
mfcc.calcMFCC(signal, sample_rate=16000, win_length=0.025,
              win_step=0.01, filters_num=26, NFFT=512, 
              low_freq=0, high_freq=None, pre_emphasis_coeff=0.97,
              cep_lifter=22, append_energy=True, append_delta=False)
  Arguments:
     signal: 1-D numpy array.
     sample_rate: Sampling rate. Defaulted to 16KHz.
     win_length: Window length. Defaulted to 0.025, which is 25ms/frame.
     win_step: Interval between the start points of adjacent frames. Defaulted to 0.01, which is 10ms.
     filters_num: Numbers of filters. Defaulted to 26.
     NFFT: Size of FFT. Defaulted to 512.
     low_freq: Lowest frequency.
     high_freq: Highest frequency.
     pre_emphasis_coeff: Coefficient for pre-emphasis. Pre-emphasis increase the energy of signal at higher frequency. Defaulted to 0.97.
     cep_lifter: Numbers of lifter for cepstral. Defaulted to 22.
     append_energy: Whether to append energy. Defaulted to True.
     append_delta: Whether to append delta to feature. Defaulted to False.
  Returns:
     2-D numpy array with shape (NUMFRAMES, features). Each frame containing filters_num of features.
</pre>

Run the following command for Fbank feature:
<pre>
mfcc.calcFbank(signal, sample_rate=16000, win_length=0.025,
               win_step=0.01, filters_num=26, NFFT=512, 
               low_freq=0, high_freq=None, pre_emphasis_coeff=0.97,
               append_energy=True, append_delta=False):
  Arguments:
     signal: 1-D numpy array.
     sample_rate: Sampling rate. Defaulted to 16KHz.
     win_length: Window length. Defaulted to 0.025, which is 25ms/frame.
     win_step: Interval between the start points of adjacent frames. Defaulted to 0.01, which is 10ms.
     filters_num: Numbers of filters. Defaulted to 26.
     NFFT: Size of FFT. Defaulted to 512.
     low_freq: Lowest frequency.
     high_freq: Highest frequency.
     pre_emphasis_coeff: Coefficient for pre-emphasis. Pre-emphasis increase the energy of signal at higher frequency. Defaulted to 0.97.
     append_energy: Whether to append energy. Defaulted to True.
     append_delta: Whether to append delta to feature. Defaulted to False.
  Returns:
     2-D numpy array with shape (NUMFRAMES, features). Each frame containing filters_num of features.
</pre>


