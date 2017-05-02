<?xml version="1.0" encoding="UTF-8"?>

<!-- Created by Clement on 090524 -->

<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform" xmlns:mb="http://musicbrainz.org/ns/mmd-1.0#">
<xsl:template match="/">
<RESULT>
    <xsl:for-each select="*[local-name()='metadata']/*[local-name()='artist-list']/*[local-name()='artist']">
        
        		 <xsl:text>&#10;</xsl:text>
                 <RECORD>
                     <xsl:text>&#10; &#32;</xsl:text>  <ITEM ANGIE-VAR='?artistname'><xsl:value-of select="mb:name"/></ITEM>
                     <xsl:text>&#10; &#32;</xsl:text>  <ITEM ANGIE-VAR='?artistid'><xsl:value-of select="@id"/></ITEM>
                      <xsl:text>&#10; &#32;</xsl:text> <ITEM ANGIE-VAR='?begindate' ><xsl:value-of select="mb:life-span/@begin"/></ITEM>
                      <xsl:text>&#10; &#32;</xsl:text> <ITEM ANGIE-VAR='?enddate'><xsl:value-of select="mb:life-span/@end"/></ITEM>
                     
                </RECORD>
               
    </xsl:for-each>  
</RESULT>
</xsl:template>
</xsl:stylesheet>
