# Video data

Video files used locally are intentionally gitignored. Store test footage under
`data/raw/`; generated or curated derivatives should go under an appropriately
named subdirectory and must remain out of commits unless licensing and repository
policy explicitly allow otherwise.

For lawful test footage, use a clip you recorded yourself or a clearly licensed
source such as [Pexels Videos](https://www.pexels.com/videos/) or
[Pixabay Videos](https://pixabay.com/videos/). Check the current license and
attribution requirements before using or redistributing any clip.

Use descriptive names that include the location/view and date, for example
`store-entrance-2026-10-08.mp4`. Keep the original file unchanged; note its
source, license, and any preprocessing in local project documentation rather
than committing footage.

Unit tests do not require a downloaded video: pytest creates short synthetic
videos in temporary directories.
