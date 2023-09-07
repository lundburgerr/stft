
#include "stft.h"

#include <math.h>

#include "kiss_fft/kiss_fft.h"
#include "signal_processing/buffer.h"

void hann_window(float* window, int length) {
  for (int i = 0; i < length; ++i) {
    window[i] = 0.5 * (1.0 - cosf(2.0 * M_PI * i / (length - 1.0)));
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

void stft_buffer_update(const float* x_real, const float* x_imag,
                        StftState* state) {
  double_buffer_complex_interleaved_update(x_real, x_imag, state->hop_size,
                                           state->buffer);
}

void stft(StftState* state) {
  // Copy values from buffer and apply window to input.
  int start = state->buffer->oldest;  // We can use this since we assume
                                      // double_buffer is of size nfft.
  for (int i = 0; i < state->nfft; ++i) {
    state->in[i].r =
        state->buffer->buffer_interleaved[2 * i + start] * state->window[i];
    state->in[i].i =
        state->buffer->buffer_interleaved[2 * i + start + 1] * state->window[i];
  }

  // Perform fft.
  kiss_fft(state->cfg, state->in, state->out);
}

void stft_get_out(const StftState* state, float* x_stft_real,
                  float* x_stft_imag) {
  int output_size = state->nfft;
  for (int i = 0; i < output_size; ++i) {
    x_stft_real[i] = state->out[i].r;
    x_stft_imag[i] = state->out[i].i;
  }
}

// Frees up memory allocated by STFT state.
void stft_free(StftState* state) {
  free(state->in);
  free(state->out);
  free(state->cfg);
}