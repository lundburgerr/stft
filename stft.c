
#include "stft.h"

#include <math.h>

#include "kiss_fft/kiss_fft.h"
#include "signal_processing/buffer.h"

void hann_window(float* window, int length) {
  for (int i = 0; i < length; ++i) {
    window[i] = 0.5 * (1.0 - cosf(2.0 * M_PI * i / (length - 1.0)));
  }
}

void calculate_synthesis_window(const float* analysis_window, int nfft,
                                int hop_size, float* synthesis_window) {
  int m = nfft / (2 * hop_size);

  // Initialize synthesis_window with values from analysis_window.
  for (int i = 0; i < nfft; i++) {
    synthesis_window[i] = analysis_window[i];
  }

  // Sum of parts as window moves in from the left of the frame.
  for (int i = 0; i < m; i++) {
    int num_samples_overlap = (i + 1) * hop_size;
    for (int j = 0; j < num_samples_overlap; j++) {
      synthesis_window[j] += analysis_window[nfft - num_samples_overlap + j];
    }
  }

  // Sum of parts as window moves out in from the right.
  for (int i = 0; i < m; i++) {
    int num_samples_overlap = (i + 1) * hop_size;
    for (int j = 0; j < num_samples_overlap; j++) {
      synthesis_window[nfft - num_samples_overlap + j] += analysis_window[j];
    }
  }

  // We calculated the sum of all overlapping windows in the frame. The
  // synthesis window is now just the inverse of that.
  for (int i = 0; i < nfft; i++) {
    synthesis_window[i] = 1.0f / synthesis_window[i];
  }

  // Flip the synthesis window so that the last hop_size samples are the first
  // etc.
  for (int i = 0; i < nfft / hop_size; i++) {
    for (int j = 0; j < hop_size; j++) {
      float temp = synthesis_window[i * hop_size + j];
      synthesis_window[i * hop_size + j] =
          synthesis_window[nfft - (i + 1) * hop_size + j];
      synthesis_window[nfft - (i + 1) * hop_size + j] = temp;
    }
  }
}

void stft_init(const float* window, DoubleBufferComplexInterleaved* buffer,
               int nfft, int hop_size, StftState* state) {
  state->nfft = nfft;
  state->hop_size = hop_size;
  state->window = window;
  state->cfg = kiss_fft_alloc(nfft, 0, NULL, NULL);
  state->buffer = buffer;

  // By default, kiss_fft uses floats as their data type.
  state->in = (kiss_fft_cpx*)malloc(sizeof(kiss_fft_cpx) * nfft);
  state->out = (kiss_fft_cpx*)malloc(sizeof(kiss_fft_cpx) * nfft);
}

void stft_inverse_init(const float* window,
                       DoubleBufferComplexInterleaved* buffer, int nfft,
                       int hop_size, StftState* state) {
  state->nfft = nfft;
  state->hop_size = hop_size;
  state->window = window;
  state->cfg = kiss_fft_alloc(nfft, 1, NULL, NULL);
  state->buffer = buffer;  // Size of this buffer is nfft * nfft / hop_size.

  // By default, kiss_fft uses floats as their data type.
  state->in = (kiss_fft_cpx*)malloc(sizeof(kiss_fft_cpx) * nfft);
  state->out = (kiss_fft_cpx*)malloc(sizeof(kiss_fft_cpx) * nfft);
}

void stft_buffer_update(const float* x_real, const float* x_imag,
                        StftState* state) {
  double_buffer_complex_interleaved_update(x_real, x_imag, state->hop_size,
                                           state->buffer);
}

void stft(StftState* state) {
  // Copy values from buffer and apply window to input.
  int start = state->buffer->oldest;  // We can use this since we assume
                                      // double_buffer is of size nfft.
  const float* analysis_window = state->window;
  for (int i = 0; i < state->nfft; ++i) {
    state->in[i].r =
        state->buffer->buffer_interleaved[2 * i + start] * analysis_window[i];
    state->in[i].i = state->buffer->buffer_interleaved[2 * i + start + 1] *
                     analysis_window[i];
  }

  // Perform fft.
  kiss_fft(state->cfg, state->in, state->out);
}

// Copy over input values to a kiss_fft_cpx array and perform inverse fft.
void stft_inverse(StftState* state, float* x_stft_real, float* x_stft_imag) {
  for (int i = 0; i < state->nfft; ++i) {
    state->in[i].r = x_stft_real[i];
    state->in[i].i = x_stft_imag[i];
  }

  // Perform inverse fft.
  kiss_fft(state->cfg, state->in, state->out);

  // Fill the double buffer with the inverse STFT of this frame.
  double_buffer_complex_interleaved_update2((float*)state->out, state->nfft,
                                            state->buffer);
}

void stft_get_out(const StftState* state, float* x_real, float* x_imag) {
  int output_size = state->nfft;
  for (int i = 0; i < output_size; ++i) {
    x_real[i] = state->out[i].r;
    x_imag[i] = state->out[i].i;
  }
}

// Adds together the nfft / hop_size overlapping frames and applies the
// synthesis window.
void stft_inverse_get_out(const StftState* state, float* x_real,
                          float* x_imag) {
  int nfft = state->nfft;
  int hop_size = state->hop_size;
  float* double_buffer = state->buffer->buffer_interleaved;
  // Compares that the result of the inverse STFT is the same as the original
  // input to STFT.buffer_interleaved;
  int start = state->buffer->oldest;
  const float* synthesis_window = state->window;

  int num_overlaps = nfft / hop_size;
  int window_start = 0;
  int frame_start = start;
  for (int i = 0; i < num_overlaps; ++i) {
    for (int j = 0; j < hop_size; ++j) {
      x_real[j] += double_buffer[2 * j + frame_start] *
                   synthesis_window[window_start + j];
      x_imag[j] += double_buffer[2 * j + frame_start + 1] *
                   synthesis_window[window_start + j];
    }
    frame_start += 2 * (nfft - hop_size);
    window_start += hop_size;
  }

  // Scale down the output for perfect reconstruction.
  float scale = 1.0f / (float)nfft;
  for (int j = 0; j < hop_size; ++j) {
    x_real[j] *= scale;
    x_imag[j] *= scale;
  }
}

// Frees up memory allocated by STFT state.
void stft_free(StftState* state) {
  free(state->in);
  free(state->out);
  free(state->cfg);
  kiss_fft_cleanup();
}