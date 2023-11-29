<script lang="ts">
  import { lcmLiveStatus, LCMLiveStatus, streamId } from '$lib/lcmLive';
  import { getPipelineValues } from '$lib/store';

  import Button from '$lib/components/Button.svelte';
  import Floppy from '$lib/icons/floppy.svelte';
  import { snapImage } from '$lib/utils';

  $: isLCMRunning = $lcmLiveStatus !== LCMLiveStatus.DISCONNECTED;
  $: console.log('isLCMRunning', isLCMRunning);
  let imageEl: HTMLImageElement;
  async function takeSnapshot() {
    if (isLCMRunning) {
      await snapImage(imageEl, {
        prompt: getPipelineValues()?.prompt,
        negative_prompt: getPipelineValues()?.negative_prompt,
        seed: getPipelineValues()?.seed,
        guidance_scale: getPipelineValues()?.guidance_scale
      });
    }
  }
  const fetchDynamicFilename = async () => {
    try {
      let dynamicFilename = ""
      const response = await fetch('http://127.0.0.1:7860/get_dynamic_filename');
      const data = await response.json();
      console.log(data)

      if (data.dynamic_filename) {
        dynamicFilename = data.dynamic_filename.dynamic_filename;
        console.log("dynamicFilename", dynamicFilename)
        const a = document.createElement("a");
        a.href = "http://127.0.0.1:7860/download/"+dynamicFilename;
        a.download = `lcm_txt_2_img${Date.now()}.gif`;
        a.click();
      }
    } catch (error) {
      console.error(error);
    }
  };
</script>

<div
  class="relative mx-auto aspect-square max-w-lg self-center overflow-hidden rounded-lg border border-slate-300"
>
  <!-- svelte-ignore a11y-missing-attribute -->
  {#if isLCMRunning}
    <img bind:this={imageEl} class="aspect-square w-full rounded-lg" src={'/stream/' + $streamId} />
    <div class="absolute bottom-1 right-1">
      <Button
        on:click={fetchDynamicFilename}
        disabled={!isLCMRunning}
        title={'Take Snapshot'}
        classList={'text-sm ml-auto text-white p-1 shadow-lg rounded-lg opacity-50'}
      >
        <Floppy classList={''} />
      </Button>
    </div>
  {:else}
    <img
      class="aspect-square w-full rounded-lg"
      src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
    />
  {/if}
</div>
