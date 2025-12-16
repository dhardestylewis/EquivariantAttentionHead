## 2025-12-13 14:29 - Organizing Paper Directories

> move both RoFormer into its own directory
> 
> create .txt equivalents for each, each in their own directories, give these directories the '.d' file extension
## 2025-12-13 14:30 - Metadata & TODO Check

> Are you keeping TODOs of everything I have asked thus far? Review this conversation as well as the PROMPTS-LOGS for anything not yet done
> 
> What I see is robust-rotary* has not yet been reorganized and renamed

## 2025-12-13 14:32 - Naming Scheme Standardization

> each of the references must be kept separately
> 
> ensure each of the directories for reference papers are named using a similar scheme to Schenck

## 2025-12-13 14:35 - Verification Request

> are you confident you named those correctly check again

## 2025-12-13 14:38 - Directory Restructuring

> i still see robust-rotary-robustness.pdf and related files at the toplevel undealtwith
> 
> name the outer references the inner papers place those papers directories within papers rename to publications and create a separate directory within new references for my own naming those reference_drafts

## 2025-12-13 14:40 - Handle Corrupted References

> Qian was misplaced
> 
> if Choromanski and Elhage are corrupted then create a file CORRUPTED_REFERENCES look up their citations and place their citation in IEEE within that file, that file should belong at the top-level within publications

## 2025-12-13 14:41 - Delete Corrupted Directories

> and then having done that delete those files and their directories

## 2025-12-13 14:43 - Cleanup and Guidelines

> are bing htmls being used by any other file?
> 
> if not delete
> 
> make a GUIDELINE that subdirs of multiple files with the same filename should be within their own directories with identical filename and extension '.d' and apply to string-drafts

## 2025-12-13 14:46 - Simplify References Structure

> remove the outer references to reference_drafts

## 2025-12-13 14:54 - Continue Session

> continue beyond Qian

## 2025-12-13 14:55 - String Drafts Organization

> place string draft detailed and string draft main into their own dir for both

## 2025-12-13 14:56 - Reference Naming Convention

> instead of marking -reference mark -REFERENCE each of these references

## 2025-12-13 15:38 - Analyze Source Files

> is this test_implementation file of string or robustness? what about cifar and mnist?

## 2025-12-13 15:45 - Source Restructuring

> move analysis movel and rotations into their own appropriately named subdir

## 2025-12-15 18:05 - Wrap Up

> warp up

## 2025-12-15 19:29 - Scientific Framing Update

> Let us set a guideline for how we discuss this with each other that we follow each and every time. When asking if works against STRING and the aspect of STRING which we wish to vary in order to demonstrate a property of STRING makes the object we are dealing with what we have temporarily called 'ESPR', then we are to continue to reference it as STRING. We will begin to iteratively reduce reliance on the language and name ESPR in all documents we touch in doing so as we formalize and document just these two results about STRING
>
> Not even framed, going forward the only aspect of ESPR we wish to preserve except as references saved into the references folder are those that are precisely these two robustness results of STRING that we are proving and providing empirical results for here - consider both my previous prompt and this one in how you revise your interpretation of how you communicate with me during this session and reflecting all this back to the GUIDELINES and the PROMPTS_LOGS

## 2025-12-15 20:06 - Code Strategy and Proof Structure

> Let's proceed in this order: leave all files as they stand without editing existing files. copy string detailed proof into a top level file. rename that new file -MONOLITH. while keeping all STRING, these specific robustness results and all machinery necessary to build them, remove any and all ESPR-related proofs. then create a non-monolithic copy as three files: one which builds string but not espr or any variant of string, another which builds the first robustness results, and the last which build the second robustness result. the first and second robustness results files should import STRING by reference from the STRING MONOLITH file. if the later robustness result depends on any or substantial machinery from the former robustness result then it should import that machinery, otherwise it should be independently formulated directly from STRING
> 
> we are not editing Schenck instead our report only presents a written out proof of these robustness results using Schenck as the starting point rather than our independent derivation of STRING
> 
> evaluate for me whether it would be faster and fewer iterations to cut down on Benchmarking Robust or instead to build up from scratch

## 2025-12-15 20:23 - Pruning the Monolith

> not just mentioneds of ESPR but any aspects of the proof beyond STRING which do not have to do with these specific robustness results. for now keep a copy of the MONOLITH you just created for -REFERENCE copy and cut away at a version of the MONOLITH for that purpose

## 2025-12-15 22:26 - Verification of Reporting Strategy

> ensure my PROMPTS-LOGS are up to date, especially what I stated regarding writing up just the proof using Schenck as a jumping off point rather than rewriting Schenck in any way. This means that we are not writing the proof present in STRING MONOLITH whatsoever only those robustness extensions
