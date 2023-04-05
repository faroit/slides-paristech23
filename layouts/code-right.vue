<!--
  Usage:
```md
---
layout: two-cols
---
# Left
This shows on the left
::right::
# Right
This shows on the right
```
-->

<script setup lang="ts">
const props = defineProps({
    class: {
        type: String,
    },
});
</script>

<template>
    <div class="grid grid-cols-2 w-full h-full">
        <div class="slidev-layout code-right col-left" :class="props.class">
            <slot />
        </div>
        <div class="col-right w-full w-full dark" :class="props.class">
            <slot name="right" />
        </div>
    </div>
</template>

<style scoped>
.col-right {
    color: #fff;
    background-color: #333333;
}
</style>